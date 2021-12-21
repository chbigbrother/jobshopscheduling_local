from django.db import models

# Create your models here.
class Information(models.Model):
    comp_id = models.IntegerField(primary_key=True)  # 작업호기
    comp_name = models.CharField(max_length=150, null=False)  # 제품명
    facility_count = models.IntegerField(null=False)  # 작업호기
    textile_type = models.CharField(max_length=50, null=False)  # 소재타입
    class Meta:
        verbose_name_plural = '회사정보등록'

    def __str__(self):
        return f'{self.comp_id} / {self.comp_name} / {self.facility_count}'

class DailyFacility(models.Model):
    idx = models.AutoField(primary_key=True) # 번호
    comp_id = models.ForeignKey("Information", related_name="dailyfacility", on_delete=models.CASCADE, db_column="comp_id") # 회사아이디
    work_start_date = models.CharField(max_length=14, null=False) # 작업시작일시
    work_end_date = models.CharField(max_length=14, null=False) # 작업종료일시
    fac_code = models.IntegerField(null=False) # 작업호기
    prod_name = models.CharField(max_length=50, null=False) # 제품명
    rpm = models.CharField(max_length=12, null=False) # rpm
    uptime = models.CharField(max_length=4, null=False) # 가동시간
    running_time = models.CharField(max_length=11, null=False) # 운행시간
    prod_output = models.CharField(max_length=11, null=False) # 생산량(yd)
    prod_rate = models.CharField(max_length=4, null=False) # 가동율
    created_at = models.DateTimeField(auto_now_add=True) # 생성일자
    modified_at = models.DateTimeField(auto_now=True) # 수정일자

class Schedule(models.Model):
    sch_id = models.CharField(max_length=50, primary_key=True)
    comp_id = models.ForeignKey("Information", related_name="company", on_delete=models.CASCADE, db_column="comp_id")
    count = models.IntegerField(null=False)
    # order_id = models.ForeignKey("order.OrderList", on_delete=models.CASCADE, db_column="order_id")
    order_id = models.CharField(max_length=120)
    sch_color = models.CharField(max_length=50, null=False)
    x_axis_1 = models.FloatField(null=False)
    y_axis_1 = models.FloatField(null=False)
    x_axis_2 = models.FloatField(null=False)
    y_axis_2 = models.FloatField(null=False)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)