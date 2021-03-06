# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2020-04-10 17:04
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('forecast_app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='QuantileDistribution',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('quantile', models.FloatField()),
                ('value_i', models.IntegerField(null=True)),
                ('value_f', models.FloatField(null=True)),
                ('value_d', models.DateField(null=True)),
                ('forecast', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='forecast_app.Forecast')),
                ('target', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='forecast_app.Target')),
                ('unit', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='forecast_app.Unit')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
