import os
import nltk
import json
from tqdm import tqdm
from textblob import Word

lemma = nltk.wordnet.WordNetLemmatizer()


def singularize(word):
    singular_form = Word(word).singularize()
    return singular_form


def combine_coco_captions(annotation_path, mscoco_version='2017'):
    if not os.path.exists(f'{annotation_path}/captions_val{mscoco_version}.json'):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists(f'{annotation_path}/captions_train{mscoco_version}.json'):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open(f'{annotation_path}/captions_val{mscoco_version}.json'))
    train_caps = json.load(open(f'{annotation_path}/captions_train{mscoco_version}.json'))
    all_caps = {
        'info': train_caps['info'],
        'licenses': train_caps['licenses'],
        'images': val_caps['images'] + train_caps['images'],
        'annotations': val_caps['annotations'] + train_caps['annotations']
    }

    return all_caps


def combine_coco_instances(annotation_path, mscoco_version='2017'):
    if not os.path.exists(f'{annotation_path}/instances_val{mscoco_version}.json'):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists(f'{annotation_path}/instances_train{mscoco_version}.json'):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open(f'{annotation_path}/instances_val{mscoco_version}.json'))
    train_instances = json.load(open(f'{annotation_path}/instances_train{mscoco_version}.json'))
    all_instances = {
        'info': train_instances['info'],
        'licenses': train_instances['licenses'],
        'type': train_instances['licenses'],
        'categories': train_instances['categories'],
        'images': train_instances['images'] + val_instances['images'],
        'annotations': val_instances['annotations'] + train_instances['annotations']
    }

    return all_instances


class CHAIR(object):

    def __init__(self, img_ids, coco_path, coco_version='2017'):
        """

        :param img_ids: 需要评估的 image_id
        :param coco_path: coco json 文件路径
        """
        self.img_id_to_objects = {img_id: [] for img_id in img_ids}
        self.coco_path = coco_path
        self.coco_version = coco_version
        # read in synonyms
        # 读取同义词
        synonyms = open('data/synonyms.txt').readlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = []

        # 将 object 的同义词映射到预设的 object，例如 puppy, canine, ... -> dog
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # common 'double words' in MSCOCO that should be treated as a single word
        # MSCOCO 中常见的“双词”应被视为单个单词
        coco_double_words = [
            'motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
            'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case',
            'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog',
            'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie',
            'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track'
        ]

        # qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        # 修饰词可能会引发误判，例如 “baby bird”（小鸟）可能会错误地把 “baby” 归为 person
        animal_words = [
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal',
            'cub'
        ]

        # qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        # 修饰词可能会引发误判，例如“passenger train”（客运列车）可能会错误地把 “passenger” 归为 person
        vehicle_words = ['jet', 'train']

        # double_word_dict will map double words to the word they should be treated as in our analysis
        # 将双字单词映射为应该处理的单词，大部分是按照原单词处理
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

        for animal_word in animal_words:
            self.double_word_dict['baby %s' % animal_word] = animal_word
            self.double_word_dict['adult %s' % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' % vehicle_word] = vehicle_word

    def _load_generated_captions_into_evaluator(self, cap_file):
        self.caption_to_eval, img_ids = load_captions_to_eval(cap_file)

        assert img_ids == set(self.img_id_to_objects.keys())

    def caption_to_words(self, caption):
        """

        :param caption:
        :return: [
                    caption 中属于 MSCOCO 的 object 单词（包括同义词）,
                    caption 中属于 MSCOCO 的 object 单词（不包括同义词）,
                    caption 中属于 MSCOCO 的 object 单词（包括同义词）在 caption 分词结果中的下标,
                    caption 的分词结果
                ]
        """
        # 对句子进行分词，单词小写，复数变单数
        words = nltk.word_tokenize(caption.lower())
        words = [singularize(w) for w in words]

        # replace double words
        # 替换所有双字单词
        i = 0
        double_words = []
        word_indexes = []
        while i < len(words):
            word_indexes.append(i)
            double_word = ' '.join(words[i:i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        # seat 是 chair 的同义词，防止将 toilet seat 分类为 chair
        if ('toilet' in words) & ('seat' in words):
            words = [word for word in words if word != 'seat']

        # get synonyms for all words in the caption
        # 筛选出句子中属于 MSCOCO 的 object 单词，包括其同义词
        word_indexes = [word_indexes[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]

        # 将所有的同义词替换为 MSCOCO 中的 object 单词
        coco_words = []
        for word in words:
            coco_words.append(self.inverse_synonym_dict[word])

        # return all the MSCOCO objects in the caption
        return words, coco_words, word_indexes, double_words

    def get_annotations_from_segments(self):
        """
        解析 instance.json 中标注的 object
        :return:
        """
        coco_segments = combine_coco_instances(self.coco_path, self.coco_version)
        segment_annotations = coco_segments['annotations']

        # MSCOCO 中所有 object 的 id 和对应名称，共 80 个object
        obj_id_to_name = {}
        for cat in coco_segments['categories']:
            obj_id_to_name[cat['id']] = cat['name']

        # 记录每张图片中出现的 object
        for i, annotation in enumerate(segment_annotations):
            img_id = annotation['image_id']
            if img_id in self.img_id_to_objects:
                # 将所有的同义词替换为 MSCOCO 中的 object 单词
                node_word = self.inverse_synonym_dict[obj_id_to_name[annotation['category_id']]]
                self.img_id_to_objects[img_id].append(node_word)

        for img_id in self.img_id_to_objects:
            self.img_id_to_objects[img_id] = set(self.img_id_to_objects[img_id])

    def get_annotations_from_captions(self):
        """
        解析 caption.json 中出现的 object
        :return:
        """
        coco_caps = combine_coco_captions(self.coco_path, self.coco_version)
        caption_annotations = coco_caps['annotations']

        for i, annotation in enumerate(caption_annotations):
            img_id = annotation['image_id']
            if img_id in self.img_id_to_objects:
                _, node_words, _, _ = self.caption_to_words(annotation['caption'])
                self.img_id_to_objects[img_id].update(node_words)

        for img_id in self.img_id_to_objects:
            self.img_id_to_objects[img_id] = set(self.img_id_to_objects[img_id])

    def get_annotations(self):
        self.get_annotations_from_segments()
        self.get_annotations_from_captions()

    def compute_chair(self, cap_file):
        self._load_generated_captions_into_evaluator(cap_file)

        img_id_to_objects = self.img_id_to_objects
        caption_to_eval = self.caption_to_eval

        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.

        chair_analysis_res = {'each_caption_chair_analysis': []}

        for cap_eval in tqdm(caption_to_eval, desc='each caption chair analysis -->'):
            cap = cap_eval['caption']
            img_id = cap_eval['image_id']

            # get all words in the caption, as well as corresponding node word
            coco_words_and_synonym, coco_words, indexes, raw_words = self.caption_to_words(cap)

            ground_truth_objects = img_id_to_objects[img_id]
            cap_dict = {
                'image_id': cap_eval['image_id'],
                'caption': cap,
                'mscoco_hallucinated_words': [],
                'mscoco_ground_truth_words': list(ground_truth_objects),
                'mscoco_generated_words': list(coco_words),
                'hallucination_indexes': [],
                'words': raw_words,
                'metrics': {
                    'CHAIRs': 0,
                    'CHAIRi': 0.
                }
            }

            # count hallucinated words
            coco_word_count += len(coco_words)
            hallucinated = False
            for word, coco_word, idx in zip(coco_words_and_synonym, coco_words, indexes):
                if coco_word not in ground_truth_objects:
                    hallucinated_word_count += 1
                    cap_dict['mscoco_hallucinated_words'].append((word, coco_word))
                    cap_dict['hallucination_indexes'].append(idx)
                    hallucinated = True

            num_caps += 1
            if hallucinated:
                num_hallucinated_caps += 1

            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.

            # 计算每个 caption 的 CHAIRi
            if len(coco_words_and_synonym) > 0:
                exists_hallucinated_word_cnt = len(cap_dict['mscoco_hallucinated_words'])
                word_cnt = float(len(coco_words_and_synonym))
                cap_dict['metrics']['CHAIRi'] = exists_hallucinated_word_cnt / word_cnt

            chair_analysis_res['each_caption_chair_analysis'].append(cap_dict)

        chair_s = (num_hallucinated_caps / num_caps)
        chair_i = (hallucinated_word_count / coco_word_count)

        chair_analysis_res['all_captions_chair_analysis'] = {
            'CHAIRs': chair_s,
            'CHAIRi': chair_i
        }

        return chair_analysis_res


def load_captions_to_eval(cap_file):
    caps = json.load(open(cap_file))
    try:
        img_ids = set([cap['image_id'] for cap in caps])
    except:
        raise Exception("Please check the caption file format")

    return caps, img_ids


def eval_file_chair(
        captions_to_eval_path,
        dir_to_dump='eval_result',
        coco_version='2017'
):
    _, img_ids = load_captions_to_eval(captions_to_eval_path)
    evaluator = CHAIR(img_ids, 'coco_annotations', coco_version)
    evaluator.get_annotations()
    chair_analysis_res = evaluator.compute_chair(captions_to_eval_path)
    json.dump(
        chair_analysis_res,
        open(f'{dir_to_dump}/{os.path.basename(captions_to_eval_path)}_eval_result.json', 'w'),
        indent=4
    )
    return chair_analysis_res


def eval_dir_chair(
        dir_to_eval='captions_to_eval',
        dir_to_dump='eval_result',
        chair_compare_path='chair_compare/chair_compare.json',
        coco_version='2017'
):
    chair_compare = {}
    files_to_eval = os.listdir(dir_to_eval)
    for i, file_name in enumerate(files_to_eval):
        print('\n' + '-' * 30)
        print(f'now eval : {file_name} [{i + 1}/{len(files_to_eval)}]')
        chair_analysis_res = eval_file_chair(os.path.join(dir_to_eval, file_name), dir_to_dump, coco_version)
        print(f'{file_name} eval done')
        print('-' * 30 + '\n')
        chair_compare[file_name] = chair_analysis_res['all_captions_chair_analysis']
    json.dump(chair_compare, open(chair_compare_path, 'w'), indent=4)


def eval_file_chair_example():
    eval_file_chair('./captions_to_eval/Clip.pth.json')


def eval_dir_chair_example():
    eval_dir_chair()


if __name__ == '__main__':
    eval_dir_chair_example()
