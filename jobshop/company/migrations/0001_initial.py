# Generated by Django 3.2.7 on 2021-11-18 08:50

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Information',
            fields=[
                ('comp_id', models.IntegerField(primary_key=True, serialize=False)),
                ('comp_name', models.CharField(max_length=150)),
                ('facility_count', models.IntegerField()),
                ('textile_type', models.CharField(max_length=50)),
            ],
            options={
                'verbose_name_plural': '회사정보등록',
            },
        ),
        migrations.CreateModel(
            name='Schedule',
            fields=[
                ('sch_id', models.CharField(max_length=50, primary_key=True, serialize=False)),
                ('count', models.IntegerField()),
                ('order_id', models.CharField(max_length=120)),
                ('sch_color', models.CharField(max_length=50)),
                ('x_axis_1', models.FloatField()),
                ('y_axis_1', models.FloatField()),
                ('x_axis_2', models.FloatField()),
                ('y_axis_2', models.FloatField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('modified_at', models.DateTimeField(auto_now=True)),
                ('comp_id', models.ForeignKey(db_column='comp_id', on_delete=django.db.models.deletion.CASCADE, related_name='company', to='company.information')),
            ],
        ),
        migrations.CreateModel(
            name='DailyFacility',
            fields=[
                ('idx', models.AutoField(primary_key=True, serialize=False)),
                ('work_start_date', models.CharField(max_length=14)),
                ('work_end_date', models.CharField(max_length=14)),
                ('fac_code', models.IntegerField()),
                ('prod_name', models.CharField(max_length=50)),
                ('rpm', models.CharField(max_length=12)),
                ('uptime', models.CharField(max_length=4)),
                ('running_time', models.CharField(max_length=11)),
                ('prod_output', models.CharField(max_length=11)),
                ('prod_rate', models.CharField(max_length=4)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('modified_at', models.DateTimeField(auto_now=True)),
                ('comp_id', models.ForeignKey(db_column='comp_id', on_delete=django.db.models.deletion.CASCADE, related_name='dailyfacility', to='company.information')),
            ],
        ),
    ]