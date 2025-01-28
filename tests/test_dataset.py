import unittest
import torch
import os
import tempfile
import gzip
import pickle
from torchtext.data import Field
from PIL import Image
import numpy as np

from main.dataset import SignTranslationDataset
from main.vocabulary import TextVocabulary, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN

class TestSignTranslationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary test data
        cls.temp_dir = tempfile.mkdtemp()
        cls.img_dir = os.path.join(cls.temp_dir, 'images')
        os.makedirs(cls.img_dir, exist_ok=True)
        
        # Create dummy images
        cls.create_dummy_images()
        
        # Create dummy dataset file
        cls.dataset_path = os.path.join(cls.temp_dir, 'dataset.pkl.gz')
        cls.create_dummy_dataset()
        
        # Create config
        cls.config = {
            'data': {
                'img_path': cls.img_dir + '/',
                'max_length': 5
            },
            'training': {
                'tokens': False
            }
        }
        
        # Create args
        cls.args = type('Args', (), {
            'input_size': 224,
            'resize': 256
        })()

    @classmethod
    def create_dummy_images(cls):
        # Create 5 dummy RGB images with different colors
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        for i in range(5):
            img = Image.new('RGB', (256, 256), color=colors[i])
            img.save(os.path.join(cls.img_dir, f'img_{i}.jpg'))

    @classmethod
    def create_dummy_dataset(cls):
        # Create dummy dataset with multiple samples
        dataset = {
            'sample_1': {
                'name': 'sample_1',
                'text': 'hello world',
                'imgs_path': [f'img_{i}.jpg' for i in range(3)]
            },
            'sample_2': {
                'name': 'sample_2',
                'text': 'how are you',
                'imgs_path': [f'img_{i}.jpg' for i in range(1, 4)]
            },
            'sample_3': {
                'name': 'sample_3',
                'text': 'sign language',
                'imgs_path': [f'img_{i}.jpg' for i in range(2, 5)]
            }
        }
        with gzip.open(cls.dataset_path, 'wb') as f:
            pickle.dump(dataset, f)

    def setUp(self):
        # Create dataset instance
        self.dataset = SignTranslationDataset(
            path=self.dataset_path,
            data_config=self.config,
            args=self.args,
            phase='train'
        )
        def tokenize_text(text):
              return text.split()
        # Create and set text field
        self.txt_field = Field(
            init_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN,
            tokenize=str.split,
            unk_token=UNK_TOKEN,
            batch_first=True,
            lower=True
        )
        
        # Create vocabulary
        self.txt_vocab = TextVocabulary(tokens=['hello', 'world', 'how', 'are', 'you', 'sign', 'language'])
        self.txt_field.vocab = self.txt_vocab
        self.dataset.set_txt_field(self.txt_field)
        print(self.dataset.txt_field.vocab, "txt vocab")
    def test_dataset_length(self):
        """Test if dataset length is correct"""
        self.assertEqual(len(self.dataset), 3)

    def test_get_item(self):
        """Test if __getitem__ returns correct format"""
        name, imgs, text = self.dataset[0]
        
        # Check name
        self.assertEqual(name, 'sample_1')
        
        # Check images
        self.assertIsInstance(imgs, torch.Tensor)
        self.assertEqual(imgs.shape[0], 3)  # 3 images
        self.assertEqual(imgs.shape[1], 3)  # RGB channels
        self.assertEqual(imgs.shape[2], 224)  # Height
        self.assertEqual(imgs.shape[3], 224)  # Width
        
        # Check text
        self.assertEqual(text, 'hello world')

    def test_collate_fn(self):
        """Test if collate_fn properly batches data with batch size 2"""
        # Create a second sample similar to the first one
        name1, imgs1, text1 = self.dataset[0]
        name2, imgs2, text2 = self.dataset[1]  # Reuse same sample for testing
        batch = [(name1, imgs1, text1), (name2, imgs2, text2)]  # Create batch of 2
        
        batch_dict = self.dataset.collate_fn(batch)
        
        # Check all expected keys are present
        expected_keys = ['video', 'attention_mask', 'name_batch', 'src_length',
                        'txt_input', 'txt_mask', 'labels']
        for key in expected_keys:
            self.assertIn(key, batch_dict)
        
        # Check batch dimensions
        print(batch_dict['video'].shape, "video shape")
        
        # don't need to check video shape as all the video images across the batch are in the same sequence
        self.assertEqual(batch_dict['txt_input'].shape[0], 2)  # Batch size 2
        self.assertEqual(batch_dict['txt_mask'].shape[0], 2)  # Batch size 2
        # self.assertEqual(batch_dict['decoder_input'].shape[0], 2)  # Batch size 2
        
        # # Check if decoder input is properly shifted for both samples
        # self.assertEqual(batch_dict['decoder_input'][0][0].item(), 
        #                 self.txt_field.vocab.stoi[BOS_TOKEN])
        # self.assertEqual(batch_dict['decoder_input'][1][0].item(), 
        #                 self.txt_field.vocab.stoi[BOS_TOKEN])

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files
        import shutil
        shutil.rmtree(cls.temp_dir)

if __name__ == '__main__':
    unittest.main() 