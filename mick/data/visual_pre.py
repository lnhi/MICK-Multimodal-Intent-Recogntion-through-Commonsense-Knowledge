import os
import csv
import sys
import logging
import pandas as pd
from transformers import BertTokenizer
import numpy as np
# from imblearn.over_sampling import SMOTE

__all__ = ['VisualCometDataset']

class VisualCometDataset:
    
    def __init__(self, args, base_attrs, visualcomet_source):
        
        self.logger = logging.getLogger(args.logger_name)
        self.visualcomet_source = visualcomet_source
        self.visualcomet_path = os.path.join(args.data_path, args.dataset, 'visualcomet', self.visualcomet_source)
        
        if args.text_backbone.startswith('bert'):
            self.feats = self._get_feats(args, base_attrs)
        else:
            raise Exception('Error: inputs are not supported text backbones.')

    def _get_feats(self, args, base_attrs):

        self.logger.info('Generate Visual Comet Features From ' + self.visualcomet_source + ' Begin...')

        processor = DatasetProcessor()

        train_xBefore_examples, train_xAfter_examples = processor.get_examples(self.visualcomet_path, 'train', self.visualcomet_source)
        train_feats = self._get_bert_feats(args, train_xBefore_examples, train_xAfter_examples, base_attrs)

        # X_train = []
        # y_train = []
        # for i, xBefore in enumerate(train_feats['xBefore']):
        #     xAfter = train_feats['xAfter'][i]
        #     # Gộp đặc trưng từ xBefore và xAfter
        #     feature = np.hstack((xBefore[0], xBefore[1], xAfter[0], xAfter[1]))  # input_ids + input_mask
        #     X_train.append(feature)
        #     # Lấy nhãn từ GUID (giả định nhãn nằm trong GUID của examples)
        #     label = int(train_xBefore_examples[i].guid.split('-')[1])
        #     y_train.append(label)

        # X_train = np.array(X_train)
        # y_train = np.array(y_train, dtype=int)

        # # Áp dụng SMOTE để cân bằng dữ liệu
        # smote = SMOTE(random_state=42)
        # X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # # Log lại phân phối nhãn sau SMOTE
        # from collections import Counter
        # self.logger.info(f"Label distribution after SMOTE: {Counter(y_resampled)}")

        # # Chuyển đổi lại dữ liệu sau SMOTE về định dạng train_feats
        # train_feats_resampled = {
        #     'xBefore': [],
        #     'xAfter': []
        # }
        # for feature, label in zip(X_resampled, y_resampled):
        #     # Tách lại input_ids và input_mask cho xBefore và xAfter
        #     input_ids_xBefore = feature[:len(train_feats['xBefore'][0][0])]
        #     input_mask_xBefore = feature[len(train_feats['xBefore'][0][0]):len(train_feats['xBefore'][0][0]) + len(train_feats['xBefore'][0][1])]
        #     input_ids_xAfter = feature[-len(train_feats['xAfter'][0][0]) - len(train_feats['xAfter'][0][1]):-len(train_feats['xAfter'][0][1])]
        #     input_mask_xAfter = feature[-len(train_feats['xAfter'][0][1]):]

        #     train_feats_resampled['xBefore'].append([input_ids_xBefore, input_mask_xBefore, [0] * len(input_ids_xBefore)])
        #     train_feats_resampled['xAfter'].append([input_ids_xAfter, input_mask_xAfter, [0] * len(input_ids_xAfter)])

        dev_xBefore_examples, dev_xAfter_examples = processor.get_examples(self.visualcomet_path, 'dev', self.visualcomet_source)
        dev_feats = self._get_bert_feats(args, dev_xBefore_examples, dev_xAfter_examples, base_attrs)

        test_xBefore_examples, test_xAfter_examples = processor.get_examples(self.visualcomet_path, 'test', self.visualcomet_source)
        test_feats = self._get_bert_feats(args, test_xBefore_examples, test_xAfter_examples, base_attrs)
        
        self.logger.info('Generate Visual Comet Features From ' + self.visualcomet_source + ' Finished...')

        return {
            'train': train_feats,
            'dev': dev_feats,
            'test': test_feats
        }

    def _get_bert_feats(self, args, xBefore_examples, xAfter_examples, base_attrs):

        max_seq_length = base_attrs["benchmarks"]['max_seq_lengths']['visualcomet']

        if args.text_backbone.startswith('bert'):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)   

        xBefore_features = convert_examples_to_features(xBefore_examples, max_seq_length, tokenizer)     
        xBefore_features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in xBefore_features]

        xAfter_features = convert_examples_to_features(xAfter_examples, max_seq_length, tokenizer)     
        xAfter_features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in xAfter_features]

        return {
            'xBefore': xBefore_features_list,
            'xAfter': xAfter_features_list
        }

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
        
    @classmethod
    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        inputs = pd.read_csv(input_file, sep=',')
        return inputs

class DatasetProcessor(DataProcessor):

    def get_examples(self, visualcomet_dir, mode, visualcomet_source):
        if mode == 'train':
            return self._create_examples(
                self._read_csv(os.path.join(visualcomet_dir, "train.csv")), "train", visualcomet_source)
        elif mode == 'dev':
            return self._create_examples(
                self._read_csv(os.path.join(visualcomet_dir, "dev.csv")), "train", visualcomet_source)
        elif mode == 'test':
            return self._create_examples(
                self._read_csv(os.path.join(visualcomet_dir, "test.csv")), "test", visualcomet_source)

    def _create_one_type_examples(self, lines, set_type, visualcomet_type, visualcomet_source):
        """Creates one relation type examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            
            if type(line) != str:
                line = 'none'

            if visualcomet_source == 'sbert':
                line = line
            elif visualcomet_source == 'comet':
                line = line[2:len(line)-2]

            if visualcomet_type == 'xBefore':
                text_a = 'Before, the speaker needed to ' + line
            elif visualcomet_type == 'xAfter':
                text_a = 'After, the speaker will most likely ' + line

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples
    
    def _create_examples(self, inputs, set_type, visualcomet_source):
        """Creates examples for the training and dev sets."""
        xBefore_examples = self._create_one_type_examples(inputs['xBefore'], set_type, 'xBefore', visualcomet_source)
        xAfter_examples = self._create_one_type_examples(inputs['xAfter'], set_type, 'xAfter', visualcomet_source)
        return xBefore_examples, xAfter_examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
        
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()