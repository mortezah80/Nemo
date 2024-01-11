import json
import nemo
import librosa
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object

an4_audio = '/home/chita/workstation/NeMo/noise3.wav'
# an4_rttm = '/home/chita/workstation/NeMo/data/google_dataset_v2/Test/sample1.rttm'
sr = 16000
signal, sr = librosa.load(an4_audio, sr=sr)


meta = {
    'audio_filepath': an4_audio,
    'offset': 0,
    'duration': None,
    'label': 'infer',
    'text': '-',
    'num_speakers': 11,
    'rttm_filepath': None,
    'uem_filepath': None
}
with open('/home/chita/workstation/NeMo/data/google_dataset_v2/Test/input_manifest.json', 'w') as fp:
    json.dump(meta, fp)
    fp.write('\n')

output_dir = '/home/chita/workstation/NeMo/new_mori/new8'

MODEL_CONFIG = '/home/chita/workstation/vad_gpu/external_apps/ai_unit/configs/diar_infer_telephonic.yaml'
config = OmegaConf.load(MODEL_CONFIG)
config.diarizer.speaker_embeddings.model_path = '/home/chita/workstation/vad_gpu/external_apps/ai_unit/models/titanet-l.nemo'
# config.model.architecture = 'resnet50'
OmegaConf.save(config, 'diar_infer_telephonic.yaml')

sd_model = ClusteringDiarizer(cfg=config)
sd_model.diarize()
