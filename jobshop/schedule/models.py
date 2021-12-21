from django.db import models
# from company.models import Company

class Schedule(models.Model):
    sch_id = models.CharField(max_length=12, primary_key=True)
    sch_color = models.CharField(max_length=50, null=False)
    x_axis_1 = models.CharField(max_length=100, null=False)
    y_axis_1 = models.CharField(max_length=100, null=False)
    x_axis_2 = models.CharField(max_length=100, null=False)
    y_axis_2 = models.CharField(max_length=100, null=False)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)