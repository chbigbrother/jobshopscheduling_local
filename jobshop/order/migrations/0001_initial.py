# Generated by Django 3.2.7 on 2021-12-23 08:14

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('company', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='OrderList',
            fields=[
                ('order_id', models.CharField(max_length=200, primary_key=True, serialize=False)),
                ('cust_name', models.CharField(max_length=50)),
                ('sch_date', models.CharField(max_length=12)),
                ('exp_date', models.CharField(max_length=12)),
                ('amount', models.FloatField()),
                ('contact', models.CharField(max_length=120)),
                ('email', models.CharField(max_length=120)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('modified_at', models.DateTimeField(auto_now=True)),
                ('prod_id', models.ForeignKey(db_column='prod_id', on_delete=django.db.models.deletion.CASCADE, to='company.product')),
            ],
        ),
        migrations.CreateModel(
            name='OrderSchedule',
            fields=[
                ('idx', models.AutoField(primary_key=True, serialize=False)),
                ('use_yn', models.CharField(max_length=1)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('modified_at', models.DateTimeField(auto_now=True)),
                ('order_id', models.ForeignKey(db_column='order_id', on_delete=django.db.models.deletion.CASCADE, related_name='orderlist', to='order.orderlist')),
                ('sch_id', models.ForeignKey(db_column='sch_id', on_delete=django.db.models.deletion.CASCADE, to='company.schedule')),
            ],
        ),
    ]
