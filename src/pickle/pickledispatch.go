package pickle

import (
	"encoding/binary"
	"fmt"
	"reflect"
	"runtime"
)

// See: https://github.com/python/cpython/blob/main/Lib/pickle.py
// See: https://docs.python.org/3/library/struct.html#format-characters

const (
	// This is the highest protocol number we know how to read.
	HIGHEST_PROTOCOL byte = 5

	MaxUint = ^uint32(0)
	MaxInt  = int(MaxUint >> 1)

	MAXSIZE = MaxInt

	PROTO byte = 0x80 // identify pickle protocol

	EMPTY_DICT      byte = '}'    // push empty dict
	BINPUT          byte = 'q'    //   "     "    "   "   " ;   "    " 1-byte arg
	MARK            byte = '('    // push special markobject on stack
	BINUNICODE      byte = 'X'    //   "     "       "  ; counted UTF-8 string argument
	GLOBAL          byte = 'c'    // push self.find_class(modname, name); 2 string args
	BININT          byte = 'J'    // push four-byte signed int
	BINSTRING       byte = 'T'    // push string; counted binary string argument
	TUPLE           byte = 't'    // build tuple from topmost stack items
	BINPERSID       byte = 'Q'    //  "       "         "  ;  "  "   "     "  stack
	BININT1         byte = 'K'    // push 1-byte unsigned int
	BININT2         byte = 'M'    // push 2-byte unsigned int
	TUPLE1          byte = '\x85' // build 1-tuple from stack top
	TUPLE2          byte = '\x86' // build 2-tuple from two topmost stack items
	TUPLE3          byte = '\x87' // build 3-tuple from three topmost stack items
	NEWTRUE         byte = '\x88' // push True
	NEWFALSE        byte = '\x89' // push False
	EMPTY_TUPLE     byte = ')'    // push empty tuple
	REDUCE          byte = 'R'    // apply callable to argtuple, both on stack
	BINGET          byte = 'h'    //   "    "    "    "   "   "  ;   "    " 1-byte arg
	LONG_BINPUT     byte = 'r'    //   "     "    "   "   " ;   "    " 4-byte arg
	STOP            byte = '.'    // every pickle ends with STOP
	SHORT_BINSTRING byte = 'U'    //  "     "   ;    "      "       "      " < 256 bytes
	SETITEMS        byte = 'u'    // modify dict by adding topmost key+value pairs
)

type dispatchFunc = func(*PickleReader) error

var dispatcher = make(map[byte]dispatchFunc)

func init() {
	dispatcher[PROTO] = load_proto
	dispatcher[EMPTY_DICT] = load_empty_dictionary
	dispatcher[BINPUT] = load_binput
	dispatcher[MARK] = load_mark
	dispatcher[BINUNICODE] = load_binunicode
	dispatcher[GLOBAL] = load_global
	dispatcher[BININT] = load_binint
	dispatcher[BINSTRING] = load_binstring
	dispatcher[TUPLE] = load_tuple
	dispatcher[BINPERSID] = load_binpersid
	dispatcher[BININT1] = load_binint1
	dispatcher[BININT2] = load_binint2
	dispatcher[TUPLE1] = load_tuple1
	dispatcher[TUPLE2] = load_tuple2
	dispatcher[TUPLE3] = load_tuple3
	dispatcher[NEWTRUE] = load_true
	dispatcher[NEWFALSE] = load_false
	dispatcher[EMPTY_TUPLE] = load_empty_tuple
	dispatcher[REDUCE] = load_reduce
	dispatcher[BINGET] = load_binget
	dispatcher[LONG_BINPUT] = load_long_binput
	dispatcher[STOP] = load_stop
	dispatcher[SHORT_BINSTRING] = load_short_binstring
	dispatcher[SETITEMS] = load_setitems
}

func pop(stack []interface{}) ([]interface{}, interface{}) {
	l := len(stack)
	element := stack[l-1]
	stack = stack[:l-1]
	return stack, element
}

func dispatch(pr *PickleReader, key byte) error {
	fn, ok := dispatcher[key]
	if ok {
		//fnReflect := runtime.FuncForPC(reflect.ValueOf(fn).Pointer()).Name()
		//fmt.Printf("Func: %v\n", fnReflect)
		return fn(pr)
	}
	return fmt.Errorf("unsupported Pickle op code: 0x%X '%c'", key, key)
}

func load_proto(pr *PickleReader) error {
	proto, err := pr.ReadByte()
	if err != nil {
		return err
	}
	if proto > HIGHEST_PROTOCOL {
		return fmt.Errorf("unsupported pickle protocol: %d", proto)
	}
	pr.proto = proto
	return nil
}

func load_empty_dictionary(pr *PickleReader) error {
	pr.Append(NewPickleDict())
	return nil
}

func load_binput(pr *PickleReader) error {
	i, err := pr.ReadByte()
	if err != nil {
		return err
	}
	pr.memo[int(i)] = pr.stack[len(pr.stack)-1]
	return nil
}

func load_mark(pr *PickleReader) error {
	pr.metastack = append(pr.metastack, pr.stack)
	pr.stack = nil
	pr.stack = make([]interface{}, 0)
	return nil
}

func load_binunicode(pr *PickleReader) error {
	buf, err := pr.Read(4)
	if err != nil {
		return err
	}
	len := int(binary.LittleEndian.Uint32(buf)) // Python equivalent: unpack('<I')
	if len > MAXSIZE {
		return fmt.Errorf("unpickling: BINUNICODE exceeds system's maximum size of %d bytes", MAXSIZE)
	}
	buf, err = pr.Read(len)
	if err != nil {
		return err
	}
	pr.Append(string(buf))
	return nil
}

func load_global(pr *PickleReader) error {
	module, err := pr.ReadLine()
	if err != nil {
		return err
	}
	name, err := pr.ReadLine()
	if err != nil {
		return err
	}
	klass, err := pr.findClass(module, name)
	if err != nil {
		return err
	}
	pr.Append(klass)
	return nil
}

func load_binint(pr *PickleReader) error {
	buf, err := pr.Read(4)
	if err != nil {
		return err
	}
	pr.Append(int32(binary.LittleEndian.Uint32(buf))) // Python equivalent: unpack('<i')
	return nil
}

func load_binstring(pr *PickleReader) error {
	// Deprecated BINSTRING uses signed 32-bit length
	buf, err := pr.Read(4)
	if err != nil {
		return err
	}
	len := int(binary.LittleEndian.Uint32(buf)) // Python equivalent: unpack('<i')
	if len < 0 {
		return fmt.Errorf("unpickling: BINSTRING pickle has negative byte count")
	}
	data, err := pr.Read(len)
	if err != nil {
		return err
	}
	pr.Append(string(data))
	return nil
}

func load_tuple(pr *PickleReader) error {
	items := pop_mark(pr)
	pr.Append(items)
	return nil
}

// Return a list of items pushed in the stack after last MARK instruction.
func pop_mark(pr *PickleReader) []interface{} {
	items := pr.stack
	var element interface{}
	pr.metastack, element = pop(pr.metastack)
	pr.stack = element.([]interface{})
	return items
}

func load_binpersid(pr *PickleReader) error {
	var pid interface{}
	pr.stack, pid = pop(pr.stack)
	result, err := pr.persistentLoad(pid.([]interface{}))
	if err != nil {
		return err
	}
	pr.Append(result)
	return nil
}

func load_binint1(pr *PickleReader) error {
	val, err := pr.ReadByte()
	if err != nil {
		return err
	}
	pr.Append(val)
	return nil
}

func load_binint2(pr *PickleReader) error {
	buf, err := pr.Read(2)
	if err != nil {
		return err
	}
	pr.Append(binary.LittleEndian.Uint16(buf)) // Python equivalent: unpack('<H')
	return nil
}

func load_tuple1(pr *PickleReader) error {
	pr.stack[len(pr.stack)-1] = PickleTuple{pr.stack[len(pr.stack)-1]}
	return nil
}

func load_tuple2(pr *PickleReader) error {
	pr.stack[len(pr.stack)-2] = PickleTuple{pr.stack[len(pr.stack)-2], pr.stack[len(pr.stack)-1]}
	pr.stack = pr.stack[:len(pr.stack)-1]
	return nil
}

func load_tuple3(pr *PickleReader) error {
	pr.stack[len(pr.stack)-3] = PickleTuple{pr.stack[len(pr.stack)-3], pr.stack[len(pr.stack)-2], pr.stack[len(pr.stack)-2]}
	pr.stack = pr.stack[:len(pr.stack)-2]
	return nil
}

func load_false(pr *PickleReader) error {
	pr.Append(false)
	return nil
}

func load_true(pr *PickleReader) error {
	pr.Append(true)
	return nil
}

func load_empty_tuple(pr *PickleReader) error {
	pr.Append(make(PickleTuple, 0))
	return nil
}

func load_reduce(pr *PickleReader) error {
	var rawArgs interface{}
	pr.stack, rawArgs = pop(pr.stack)

	rawArgsArr := rawArgs.([]interface{})

	fn := pr.stack[len(pr.stack)-1]
	fnType := reflect.TypeOf(fn)
	requiredArgsCount := fnType.NumIn()

	args := make([]reflect.Value, requiredArgsCount)

	for i, arg := range rawArgsArr {
		argType := fnType.In(i)
		argVal := reflect.ValueOf(arg)
		if !argVal.CanConvert(argType) {
			fnName := runtime.FuncForPC(reflect.ValueOf(fn).Pointer()).Name()
			return fmt.Errorf("cannot convert value in type %s to type %s, for argument index %d of function \"%s\"", argVal.Type().Name(), argType.Name(), i, fnName)
		}
		args[i] = argVal.Convert(argType)
	}

	// For optional and/or remaining arguments which we don't have value,
	// we create zero/nil values in each input type
	for i := len(rawArgsArr); i < requiredArgsCount; i++ {
		argType := fnType.In(i)
		args[i] = reflect.Zero(argType)
	}

	resultArr := reflect.ValueOf(fn).Call(args)
	result := make([]interface{}, len(resultArr))
	for i, item := range resultArr {
		result[i] = item.Interface()
	}

	pr.stack[len(pr.stack)-1] = result[0]
	return nil
}

func load_binget(pr *PickleReader) error {
	i, err := pr.ReadByte()
	if err != nil {
		return err
	}
	item, ok := pr.memo[int(i)]
	if !ok {
		return fmt.Errorf("memo value not found at index %d", i)
	}
	pr.Append(item)
	return nil
}

func load_long_binput(pr *PickleReader) error {
	buf, err := pr.Read(4)
	if err != nil {
		return err
	}
	i := int(binary.LittleEndian.Uint32(buf)) // Python equivalent: unpack('<I')
	if i > MAXSIZE {
		return fmt.Errorf("negative LONG_BINPUT argument")
	}
	pr.memo[i] = pr.stack[len(pr.stack)-1]
	return nil
}

func load_stop(pr *PickleReader) error {
	var value interface{}
	pr.stack, value = pop(pr.stack)
	return &StopSignal{value.(*PickleDict)}
}

func load_short_binstring(pr *PickleReader) error {
	len, err := pr.ReadByte()
	if err != nil {
		return err
	}
	data, err := pr.Read(int(len))
	if err != nil {
		return err
	}
	pr.Append(string(data))
	return nil
}

func load_setitems(pr *PickleReader) error {
	items := pop_mark(pr)
	dict := pr.stack[len(pr.stack)-1].(*PickleDict)
	for i := 0; i < len(items); i += 2 {
		dict.Set(items[i].(string), items[i+1])
	}
	return nil
}
