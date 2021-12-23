# Generated by Django 3.2.7 on 2021-12-23 08:13

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Facility',
            fields=[
                ('facility_id', models.CharField(max_length=50, primary_key=True, serialize=False)),
                ('facility_name', models.IntegerField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('modified_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='Information',
            fields=[
                ('comp_id', models.IntegerField(primary_key=True, serialize=False)),
                ('comp_name', models.CharField(max_length=150)),
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
                ('work_str_date', models.CharField(max_length=12)),
                ('work_end_date', models.CharField(max_length=12)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('modified_at', models.DateTimeField(auto_now=True)),
                ('comp_id', models.ForeignKey(db_column='comp_id', on_delete=django.db.models.deletion.CASCADE, related_name='company_id', to='company.information')),
                ('facility_id', models.ForeignKey(db_column='facility_id', on_delete=django.db.models.deletion.CASCADE, related_name='facilityId', to='company.facility')),
            ],
        ),
        migrations.CreateModel(
            name='Product',
            fields=[
                ('prod_id', models.CharField(max_length=120, primary_key=True, serialize=False)),
                ('prod_name', models.CharField(max_length=120)),
                ('density', models.FloatField()),
                ('rpm', models.FloatField()),
                ('daily_prod_rate', models.FloatField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('modified_at', models.DateTimeField(auto_now=True)),
                ('comp_id', models.ForeignKey(db_column='comp_id', on_delete=django.db.models.deletion.CASCADE, related_name='company', to='company.information')),
            ],
        ),
        migrations.AddField(
            model_name='facility',
            name='comp_id',
            field=models.ForeignKey(db_column='comp_id', on_delete=django.db.models.deletion.CASCADE, related_name='facilitycompany', to='company.information'),
        ),
    ]
