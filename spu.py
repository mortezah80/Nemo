import sys
import yaml
import os
import torchaudio
from pydub import AudioSegment
from typing import List, Dict, Union
from vad_test import vad
import json
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
import time


class SpeechProcessingUnit:
    def __init__(self, testdata_dir, num_speakers=5, gpu_number=1, threshold=0.5):
        path = os.path.dirname(os.path.realpath(__file__))
        self.path = path
        self.num_speakers = num_speakers
        self.testdata_dir = testdata_dir
        file_name_prepare, file_duration = self.prepare(testdata_dir)
        self.file_name = file_name_prepare
        # initialize segmentation pipeline
        config_vad, config_dia = self.initialize_config(file_name_prepare, gpu_number, threshold)
        self.cfg = config_vad
        self.config_dia = config_dia

    def initialize_config(self, testdata_dir, gpu_number, threshold):

        with open(f"{self.path}/configs/vad_inference_postprocessing.yaml") as file:
            config_vad = yaml.load(file, Loader=yaml.FullLoader)

        audio_name = testdata_dir.split('/')[-1]
        config_vad["out_manifest_filepath"] = audio_name.split('.')[0] + '.json'
        config_vad["vad"]["model_path"] = f'{self.path}/models/epoch=200-step=5025.ckpt'
        config_vad["dataset"] = f'{self.path}/configs/vad_test.json'
        config_vad["vad"]["parameters"]["postprocessing"]["onset"] = float(threshold) + 0.1
        config_vad["vad"]["parameters"]["postprocessing"]["offset"] = float(threshold) - 0.1

        with open(f'{self.path}/configs/vad_inference_postprocessing.yaml', 'w') as s:
            yaml.dump(config_vad, s)

        with open(config_vad["dataset"]) as json_file:
            data_dir = json.load(json_file)

        data_dir['audio_filepath'] = testdata_dir

        with open(config_vad["dataset"], 'w') as json_file:
            json.dump(data_dir, json_file)

        with open(f"{self.path}/configs/diar_infer_telephonic.yaml") as file:
            config_dia = yaml.load(file, Loader=yaml.FullLoader)

        config_dia["diarizer"]["speaker_embeddings"]["model_path"] = f'{self.path}/models/titanet-l.nemo'
        config_dia["diarizer"]["manifest_filepath"] = f'{self.path}/configs/input_manifest.json'
        self.out_dir = f'{self.path}/speaker_diar_output'
        config_dia["diarizer"]["out_dir"] = self.out_dir
        config_dia["diarizer"]["vad"]["external_vad_manifest"] = f'{self.file_name.split(".")[0]}.json'
        config_dia["diarizer"]["vad"]["model_path"] = None
        config_dia["device"] = f"cuda:{gpu_number}"
        config_dia["diarizer"]["vad"]["parameters"]["onset"] = float(threshold) + 0.1
        config_dia["diarizer"]["vad"]["parameters"]["offset"] = float(threshold) - 0.1

        with open(f'{self.path}/configs/diar_infer_telephonic.yaml', 'w') as s:
            yaml.dump(config_dia, s)

        return config_vad, config_dia

    def __call__(self, task: str) -> Union[Dict, List, None, str]:

        if task == "vad":
            out_manifest_filepath = self.vad_task(self.cfg)
            self.out_manifest_filepath = out_manifest_filepath
            vad_out = []
            with open(f"{self.path}/{self.file_name}.json") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.split(" ")
                    start = float(line[3][:len(line[3]) - 1])
                    end = start + float(line[5][:len(line[5]) - 1])
                    dia_dict = {"begin": start, "end": end}
                    vad_out.append(dia_dict)

            with open(f"{self.path}/{self.file_name}_vad_dict.json", 'w') as fp:
                json.dump(vad_out, fp, indent=4)
                fp.write('\n')

            return out_manifest_filepath
        elif task == "diarization":
            self.diarization_task(self.file_name)
            return dia_output

        elif task == "vad_diarization":
            self.vad_task(self.cfg)
            self.diarization_task(self.file_name)
            vad_out = []
            with open(f"{self.path}/{self.file_name.split('/')[-1].split('.')[0]}.json") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(" ")
                    start = float(line[3][:len(line[3]) - 1])
                    end = start + float(line[5][:len(line[5]) - 1])
                    dia_dict = {"begin": start, "end": end}
                    vad_out.append(dia_dict)

            dia_out = []
            num_speakers = self.num_speakers
            with open(f"{self.out_dir}/pred_rttms/{self.file_name.split('/')[-1].split('.')[0]}.rttm") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(" ")
                    start = float(line[5])
                    end = start + float(line[8])
                    speaker = int(line[11][8:])
                    speakers = []
                    for i in range(num_speakers):
                        if i == speaker:
                            temp = {"id": i, "prob": 100}
                        else:
                            temp = {"id": i, "prob": 0}
                        speakers.append(temp)
                    new_list = sorted(speakers, key=lambda d: d['prob'], reverse=True)
                    dia_dict = {"begin": start, "end": end, "speakers": new_list}
                    dia_out.append(dia_dict)

            output = {
                "sad_annotation": vad_out,
                "dia_annotation": dia_out
            }
            output_dict_json = f"{self.path}/{self.testdata_dir.split('/')[-1].split('.')[0]}_out_dict.json"
            with open(output_dict_json, 'w') as fp:
                json.dump(output, fp, indent=4)
                fp.write('\n')
            del_file_json = f"{self.path}/{self.file_name.split('/')[-1].split('.')[0]}.json"
            del_file_wav = f"{self.path}/{self.file_name.split('/')[-1].split('.')[0]}.wav"

            os.remove(f'{del_file_json}')
            os.remove(f'{del_file_wav}')

            os.remove(f'{self.path}/manifest_vad_input.json')

        else:
            sys.exit("your task should be one of vad, diarization, vad_diarization")
        return output_dict_json

    def prepare(self, file_dir):
        if (file_dir.split(".")[-1]) == "mp3":
            sound = AudioSegment.from_mp3(file_dir)
            sound = sound.set_channels(1)
            sound.export(file_dir, format="mp3")
            torchaudio.set_audio_backend("sox_io")
            fs = 16000
            signal, sample_rate = torchaudio.load(file_dir, format="mp3")

            file_duration = len(signal[0]) / sample_rate
            # adjust fs
            adjust_fs = torchaudio.transforms.Resample(sample_rate, fs)
            signal = adjust_fs(signal)
            new_name = str(time.time()).split(".")[-1]
            torchaudio.save(f'{self.path}/{new_name}.wav', signal, fs, format="wav")
            file_dir = f'{self.path}/{new_name}.wav'
        else:
            torchaudio.set_audio_backend("sox_io")
            fs = 16000
            signal, sample_rate = torchaudio.load(file_dir, format="wav")
            file_duration = len(signal[0]) / sample_rate
            # adjust fs
            adjust_fs = torchaudio.transforms.Resample(sample_rate, fs)
            signal = adjust_fs(signal)
            new_name = str(time.time()).split(".")[-1]
            torchaudio.save(f'{self.path}/{new_name}.wav', signal, fs, format="wav")
            file_dir = f'{self.path}/{new_name}.wav'

        return file_dir, file_duration

    def vad_task(self, cfg):
        out_manifest_filepath = vad(cfg)
        return out_manifest_filepath

    def diarization_task(self, testdata_dir):
        meta = {
            'audio_filepath': testdata_dir,
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': self.num_speakers,
            'rttm_filepath': None,
            'uem_filepath': None
        }
        with open(f'{self.path}/configs/input_manifest.json', 'w') as fp:
            json.dump(meta, fp)
            fp.write('\n')

        model_config = f'{self.path}/configs/diar_infer_telephonic.yaml'
        config_dia = OmegaConf.load(model_config)

        sd_model = ClusteringDiarizer(cfg=config_dia)
        sd_model.diarize()


if __name__ == "__main__":
    file_path_index = -1
    save_path_index = -1
    vad_path_index = -1
    threshold_index = -1
    gpu_number_index = -1
    num_speaker_index = -1

    threshold = 0.5
    save_path = ""
    file_path = ""
    vad_path = ""
    gpu_number = 0
    num_speakers = 5

    for i, arg in enumerate(sys.argv):
        if arg == "file_path":
            file_path_index = i + 1

        elif arg == "save_path":
            save_path_index = i + 1

        elif arg == "vad_path":
            vad_path_index = i + 1

        elif arg == "threshold":
            threshold_index = i + 1

        elif arg == 'gpu_number':
            gpu_number_index = i + 1

        elif arg == "num_speakers":
            num_speaker_index = i + 1

        elif vad_path_index == i:
            vad_path = arg

        elif file_path_index == i:
            file_path = arg

        elif save_path_index == i:
            save_path = arg

        elif threshold_index == i:
            threshold = arg

        elif gpu_number_index == i:
            gpu_number = int(arg)

        elif num_speaker_index == i:
            num_speakers = int(arg)

    spu = SpeechProcessingUnit(file_path, num_speakers, gpu_number, threshold)

    task = "vad_diarization"
    output_string = spu(task)

    print("===start===")
    with open(output_string) as f:
        dict_out = json.load(f)
        print(dict_out)
    print("====end====")
    os.remove(output_string)
