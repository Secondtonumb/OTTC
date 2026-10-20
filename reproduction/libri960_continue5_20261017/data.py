"""Same Libri100 waveform normalization without a 960-hour waveform copy."""
import bootstrap
from bootstrap import HERE, ROOT
import datasets
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor


class AudioTransform:
    def __init__(self):
        self.processor=None

    def __call__(self,batch):
        # HF's length sampler asks for just this column.
        if 'wav' not in batch:return batch
        if self.processor is None:
            self.processor=Wav2Vec2Processor.from_pretrained(HERE/'data/processor')
        audio=[]
        for wav,length in zip(batch['wav'],batch['length']):
            samples,sr=sf.read(wav,dtype='float32')
            assert sr==16000 and samples.ndim==1 and abs(len(samples)-length)<=16,(wav,len(samples),length)
            values=self.processor(samples,sampling_rate=sr).input_values[0]
            assert np.isfinite(values).all()
            audio.append(values)
        return dict(input_values=audio,labels=batch['labels'],length=batch['length'])


def load_data(smoke=False):
    train=datasets.Dataset.from_file(str(HERE/'data/train_metadata.arrow'))
    if smoke:
        lengths=train['length']
        train=train.select(sorted(range(len(lengths)),key=lengths.__getitem__,reverse=True)[:64])
    train=train.with_transform(AudioTransform())
    dev=datasets.load_from_disk(HERE/'data/cache_dev')
    if smoke:dev=dev.select(range(16))
    return datasets.DatasetDict(train=train,dev=dev)


def check_normalization():
    """Check the lazy path against the actual prior cached training tensors."""
    source=datasets.Dataset.from_file(str(HERE/'data/train_metadata.arrow'))
    prior=datasets.load_from_disk(ROOT/'reproduction/libri100_singlehead_20261014/data/cache_train')
    transformed=source.with_transform(AudioTransform())
    for i in [0,1,100,1234,10000,28537]:
        assert source[i]['utt'].split('-')[0] in source[i]['wav']
        current=transformed[i];previous=prior[i]
        assert current['labels']==previous['labels']
        np.testing.assert_array_equal(current['input_values'],np.asarray(previous['input_values'],dtype='float32'))
    return dict(samples=6,lazy_audio_matches_parent_cached_normalization=True)
