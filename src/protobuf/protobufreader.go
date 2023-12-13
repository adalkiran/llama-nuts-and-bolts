package protobuf

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"unicode/utf8"
)

// See: https://github.com/protocolbuffers/protobuf-go/blob/e8baad6b6c9e2bb1c48e4963866248d4f35d4fd7/encoding/protowire/wire.go

// Number represents the field number.
type Number int32

// Type represents the wire type.
type Type int8

const (
	VarintType     Type = 0
	Fixed32Type    Type = 5
	Fixed64Type    Type = 1
	BytesType      Type = 2
	StartGroupType Type = 3
	EndGroupType   Type = 4
)

type Message struct {
	Number Number
	Value  interface{}
}

func (m Message) String() string {
	return fmt.Sprintf("%d: { %v }", m.Number, m.Value)
}

type ProtoDescriptor struct {
	MainObjectConstructorFn func() interface{}
	MessageProcessorFns     map[Number]func(interface{}, Message)
}

type ProtobufReader struct {
	fileReader      *bufio.Reader
	protoDescriptor ProtoDescriptor
}

func NewProtobufReader(fileReader io.ReadCloser, protoDescriptor ProtoDescriptor) *ProtobufReader {
	result := new(ProtobufReader)
	result.fileReader = bufio.NewReader(fileReader)
	result.protoDescriptor = protoDescriptor
	return result
}

func (pbr *ProtobufReader) Unmarshal() (mainObject interface{}, err error) {
	mainObject = pbr.protoDescriptor.MainObjectConstructorFn()
	counter := 0
	for {
		message, ok := pbr.readMessage()
		if !ok {
			break
		}

		//fmt.Printf("%d | %v\n", counter, *message)
		if pbr.protoDescriptor.MessageProcessorFns[message.Number] == nil {
			return nil, fmt.Errorf("cannot find MessageProcessorFns item for number %d", message.Number)
		}
		pbr.protoDescriptor.MessageProcessorFns[message.Number](mainObject, *message)
		counter++
	}
	return
}

func (pbr *ProtobufReader) readMessage() (message *Message, ok bool) {
	_, err := pbr.fileReader.Peek(1)
	if err != nil {
		return nil, false
	}
	number, item, ok := pbr.readField(pbr.fileReader)
	if !ok {
		return nil, false
	}
	return &Message{number, item}, true
}

func (pbr *ProtobufReader) readField(r *bufio.Reader) (number Number, result interface{}, ok bool) {
	remainingPos := r.Buffered()
	number, type_, ok := pbr.readTag(r)
	if !ok {
		return
	}
	switch type_ {
	case BytesType:
		{
			resultMap := make(map[Number]interface{})
			b, ok := pbr.readValueBytes(r)
			if !ok {
				pbr.undoRead(r, remainingPos-r.Buffered())
				return number, nil, false
			}
			localReader := bufio.NewReader(bytes.NewReader(b))
			if len(b) > 0 {
				var itemNumber Number
				var item interface{}
				allOk := false
				for {
					_, err := localReader.Peek(1)
					if err != nil {
						break
					}
					remainingPosLocal := r.Buffered()
					itemNumber, item, allOk = pbr.readField(localReader)
					if !allOk {
						pbr.undoRead(localReader, remainingPosLocal-localReader.Buffered())
						break
					}
					// A rule to prevent misinterpretation of byte arrays
					if ((len(resultMap) > 0 && int(itemNumber)/len(resultMap) > 3) ||
						(len(resultMap) == 0 && int(itemNumber) > 2)) && utf8.Valid(b) {
						pbr.undoRead(localReader, remainingPosLocal-localReader.Buffered())
						break
					}
					resultMap[itemNumber] = item
				}
			}
			_, err := localReader.Peek(1)
			if err == nil || len(b) == 0 {
				if len(b) > 0 && utf8.Valid(b) {
					return number, string(b), true
				} else {
					return number, b, true
				}
			}
			if len(resultMap) == 0 {
				pbr.undoRead(r, remainingPos-r.Buffered())
				return number, nil, false
			}
			return number, resultMap, true
		}
	case Fixed32Type:
		{
			result, ok = pbr.readValueInt32(r)
			return
		}
	case VarintType:
		{
			result, ok = pbr.readVarintSigned(r)
			return
		}
	}
	return number, nil, false
}

func (pbr *ProtobufReader) readTag(r *bufio.Reader) (number Number, type_ Type, ok bool) {
	val, ok := pbr.readVarint(r)
	if !ok {
		return 0, 0, false
	}
	number = Number(val >> 3)
	type_ = Type(val & 7)
	return
}

func (pbr *ProtobufReader) readVarint(r *bufio.Reader) (value uint64, ok bool) {
	for count := 0; count < 10; count++ {
		b, err := r.ReadByte()
		if err != nil {
			pbr.undoRead(r, count)
			return 0, false
		}
		if count == 9 && b > 1 {
			// The tenth byte has a special upper limit: it may only be 0 or 1.
			pbr.undoRead(r, count)
			return 0, false
		}
		value |= uint64(b&0x7f) << (count * 7)
		if b&0x80 == 0 {
			break
		}
	}
	return value, true
}

func (pbr *ProtobufReader) readVarintSigned(r *bufio.Reader) (value int64, ok bool) {
	val, ok := pbr.readVarint(r)
	if !ok {
		return 0, ok
	}
	return int64(val), ok
}

func (pbr *ProtobufReader) readValueBytes(r *bufio.Reader) (value []byte, ok bool) {
	len, ok := pbr.readVarint(r)
	if !ok {
		return nil, false
	}
	if uint64(r.Size()) < len {
		return nil, false
	}
	buf := make([]byte, len)
	readCount, err := io.ReadFull(r, buf)
	if err != nil || readCount != int(len) {
		pbr.undoRead(r, readCount)
		return nil, false
	}
	return buf, true
}

func (pbr *ProtobufReader) readValueInt32(r *bufio.Reader) (value float32, ok bool) {
	buf := make([]byte, 4)
	readCount, err := r.Read(buf)
	if err != nil || readCount != 4 {
		pbr.undoRead(r, readCount)
		return 0, false
	}
	bytesRepresentation := binary.LittleEndian.Uint32(buf)
	float32Value := math.Float32frombits(bytesRepresentation)

	return float32Value, true
}

func (pbr *ProtobufReader) undoRead(r *bufio.Reader, count int) error {
	for i := 0; i < count; i++ {
		err := r.UnreadByte()
		if err != nil {
			return err
		}
	}
	return nil
}
