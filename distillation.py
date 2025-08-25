import torch
import torch.nn as nn

class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

    def forward(self, x):
        # Kết quả đầu ra của teacher không cần tính gradient
        with torch.no_grad():
            teacher_predictions = self.teacher(x)

        student_predictions = self.student(x)

        return student_predictions, teacher_predictions