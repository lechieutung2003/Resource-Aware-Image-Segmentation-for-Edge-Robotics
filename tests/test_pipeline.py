import yaml
import sys
import os
import torch
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.dataset import get_dataloaders
from src.models.teacher.model import get_teacher_model
from src.models.student.model import get_student_model
from src.distillation.loss import get_distillation_loss

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('configs/config.yaml', 'r') as f:
            cls.config = yaml.safe_load(f)
            
    def test_dataset(self):
        train_loader, val_loader, test_loader = get_dataloaders(self.config)
        self.assertGreater(len(train_loader.dataset), 0)
        
        images, masks = next(iter(train_loader))
        img_size = self.config['dataset']['img_size']
        self.assertEqual(images.shape[2:], tuple(img_size))
        self.assertEqual(masks.shape[1:], tuple(img_size))
        
    def test_teacher_forward(self):
        model = get_teacher_model(self.config)
        model.eval()
        img_size = self.config['dataset']['img_size']
        dummy_input = torch.randn(1, 3, img_size[0], img_size[1])
        output = model(dummy_input)['out']
        
        num_classes = self.config['dataset']['num_classes']
        self.assertEqual(output.shape, (1, num_classes, img_size[0], img_size[1]))
        
    def test_student_forward(self):
        model = get_student_model(self.config)
        model.eval()
        img_size = self.config['dataset']['img_size']
        dummy_input = torch.randn(1, 3, img_size[0], img_size[1])
        output = model(dummy_input)['out']
        
        num_classes = self.config['dataset']['num_classes']
        self.assertEqual(output.shape, (1, num_classes, img_size[0], img_size[1]))
        
    def test_distillation_loss(self):
        criterion = get_distillation_loss(self.config)
        num_classes = self.config['dataset']['num_classes']
        img_size = self.config['dataset']['img_size']
        
        student_logits = torch.randn(2, num_classes, img_size[0], img_size[1])
        teacher_logits = torch.randn(2, num_classes, img_size[0], img_size[1])
        targets = torch.randint(0, num_classes, (2, img_size[0], img_size[1]))
        
        total_loss, ce_loss, kd_loss = criterion(student_logits, teacher_logits, targets)
        
        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isnan(ce_loss))
        self.assertFalse(torch.isnan(kd_loss))

if __name__ == '__main__':
    unittest.main()
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
