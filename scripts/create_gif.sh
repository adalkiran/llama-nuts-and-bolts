#!/bin/bash

# Note: This script was created to generate an animated gif by concatenating multiple screenshot .mov video files.
# It was used to generate animated gif file used in the README file. The original input screen recording video files weren't included in this GitHub repository.
# However, this script was included to give an idea about how it was created via FFMPEG.

scale_width=800
input_video_time_rate=0.05
fps=5
first_frame_wait_secs=2
last_frame_wait_secs=4
output_file_name="output.gif"

i=0
input_list=""
filter_cmp1=""
filter_cmp2=""
for input_mov in input_beijing_ny input_tr_flag input_quantum_cmp input_relativity; do
  echo "Taking first and last frame from ${input_mov}.mov -> temp_first_frame_${input_mov}.jpg and temp_last_frame_${input_mov}.jpg"

  i1=$((i + 1))
  i2=$((i + 2))

  ffmpeg -y -i "${input_mov}.mov" -vframes 1 "temp_first_frame_${input_mov}.jpg"
  ffmpeg -y -sseof -1 -i "${input_mov}.mov" -update 1 -q:v 1 "temp_last_frame_${input_mov}.jpg"


  input_list="${input_list}-loop 1 -t ${first_frame_wait_secs} -i temp_first_frame_${input_mov}.jpg \
      -i "${input_mov}.mov" \
      -loop 1 -t ${last_frame_wait_secs} -i temp_last_frame_${input_mov}.jpg "

  filter_cmp1="${filter_cmp1}[${i}:v]fps=${fps},scale=${scale_width}:-1:flags=lanczos,setpts=PTS-STARTPTS[part${i}]; \
[${i1}:v]fps=${fps},scale=${scale_width}:-1:flags=lanczos,setpts=${input_video_time_rate}*PTS[part${i1}]; \
[${i2}:v]fps=${fps},scale=${scale_width}:-1:flags=lanczos,setpts=PTS-STARTPTS[part${i2}]; "

  filter_cmp2="${filter_cmp2}[part${i}][part${i1}][part${i2}]"
  i=$((i + 3))
done

filter_cmp2="${filter_cmp2}concat=n=${i}:v=1:a=0[out]; \
[out]split=2[out1][out2]; [out1]palettegen[pal]; [out2][pal]paletteuse"

ffmpeg -y \
     ${input_list} \
     -filter_complex "${filter_cmp1} ${filter_cmp2}" -r ${fps} "${output_file_name}"

rm -rf ./temp_*.jpg

echo -e "\n\n\nThe command was executed:"
echo "ffmpeg -y ${input_list} -filter_complex \"${filter_cmp1} ${filter_cmp2}\" -r ${fps} \"${output_file_name}\""
