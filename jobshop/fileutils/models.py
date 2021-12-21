from django.db import models


class FileUpload(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=120, null=False)
    author = models.CharField(max_length=20, null=False)
    file = models.FileField(upload_to='')
    content = models.TextField(max_length=500, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        verbose_name_plural = '파일 게시판'

class Photo(models.Model):
    post = models.ForeignKey(FileUpload, on_delete=models.CASCADE, null=True)
    image = models.ImageField(upload_to='images/', blank=True, null=True)

class FileUploadCsv(models.Model):
    title = models.TextField(max_length=500, null=True, blank=True)
    file = models.FileField(null=True)
