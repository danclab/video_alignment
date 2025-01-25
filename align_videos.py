import glob
import json
import os
import wave

import cv2
import numpy as np
import subprocess
from align_videos_by_soundtrack.align import SyncDetector
from align_videos_by_soundtrack.utils import check_and_decode_filenames


# Resize frame
def quick_resize(data, scale, og_width, og_height):
    width = int(og_width * scale)
    height = int(og_height * scale)
    dim = (width, height)
    resized = cv2.resize(
        data,
        dim,
        interpolation=cv2.INTER_AREA
    )
    return resized


def align_two_videos(vid_1, vid_2, output_path):
    vid_1_name = os.path.splitext(os.path.split(vid_1)[1])[0]
    vid_2_name = os.path.splitext(os.path.split(vid_2)[1])[0]
    combined_video_fname = os.path.join(output_path, "combined_{}_{}.avi".format(vid_1_name, vid_2_name))
    combined_json_fname = os.path.join(output_path, "combined_{}_{}.json".format(vid_1_name, vid_2_name))

    if not os.path.exists(combined_video_fname):
        # Align videos
        file_specs = check_and_decode_filenames([vid_1, vid_2], min_num_files=2)
        with SyncDetector() as det:
            result = det.align(file_specs)
            # Writing to sample.json
            with open(combined_json_fname, "w") as outfile:
                outfile.write(json.dumps(result, indent=4))

        # Amount to trim from beginning of video 1 (seconds)
        vid1_trim=result[0]['trim']
        vid2_trim=result[1]['trim']

        # FPS
        fps = result[0]['orig_streams'][0]['fps']
        # Number of frames to trim from beginning of video 1
        if np.abs(vid1_trim)>np.abs(vid2_trim):
            offset = int(vid1_trim * fps)
        else:
            offset=-int(vid2_trim*fps)

        # Open videos
        vidcap1 = cv2.VideoCapture(vid_1)
        vidcap2 = cv2.VideoCapture(vid_2)


        # Convert vid1 audio to wav
        audio_file = os.path.join(output_path,'{}.wav'.format(os.path.split(os.path.splitext(vid_1)[0])[-1]))
        cmd = ['ffmpeg', '-y', '-i', vid_1, audio_file]
        subprocess.run(cmd)

        # Read audio data
        audio = wave.open(audio_file, mode='rb')
        audio_params = audio.getparams()
        audio_data = audio.readframes(audio_params.nframes)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)  # Audio format is usually 16 bits; buffer

        # Video 1 starts too soon
        if offset > 0:
            for i in range(offset):
                vidcap1.read()
            audio_offset = vid1_trim
        # Video 2 starts too soon
        else:
            for i in range(-offset):
                vidcap2.read()
            audio_offset = 0

        # Crop audio
        audio_part = audio_data[int(audio_offset * audio_params.framerate):]

        # Video size
        f_size = (1920, 540)

        # Write aligned video without audio
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        silent_video_fname = os.path.join(output_path, "combined_video_silent.avi")
        cam_vid = cv2.VideoWriter(
            silent_video_fname,
            fourcc,
            float(fps),
            f_size
        )

        # For each frame
        while True:
            # Read frames
            success1, image1 = vidcap1.read()
            success2, image2 = vidcap2.read()

            # Break if at least one video has ended
            if not success1 and success2:
                image1=np.ones(image2.shape).astype(np.uint8)
            elif not success2 and success1:
                image2 = np.ones(image1.shape).astype(np.uint8)
            elif not (success1 or success2):
                break

            # Resize each frame
            if image1.shape[0]==1080:
                resized1 = quick_resize(image1, 0.5, image1.shape[1], image1.shape[0])
            elif image1.shape[0]==2160:
                resized1 = quick_resize(image1, 0.25, image1.shape[1], image1.shape[0])
            else:
                resized1 = image1
            if image2.shape[0]==1080:
                resized2 = quick_resize(image2, 0.5, image2.shape[1], image2.shape[0])
            elif image2.shape[0]==2160:
                resized2 = quick_resize(image2, 0.25, image2.shape[1], image2.shape[0])
            else:
                resized2 = image2

            # Show camera images side by side
            data = np.hstack([resized1, resized2])
            cam_vid.write(data[:, :, :3])

        cam_vid.release()

        # Write cropped audio to file
        fp = wave.Wave_write(audio_file)
        # Set the parameters of fp
        fp.setframerate(audio_params.framerate)
        fp.setnframes(len(audio_part))
        fp.setnchannels(audio_params.nchannels)
        fp.setsampwidth(audio_params.sampwidth)
        fp.writeframes(audio_part.tobytes())
        fp.close()

        # Combine video and audio
        cmd = ['ffmpeg', '-i', audio_file, '-i', silent_video_fname, '-q:v', '0', combined_video_fname]
        subprocess.run(cmd)

        # Remove temporary files
        os.remove(audio_file)
        os.remove(silent_video_fname)