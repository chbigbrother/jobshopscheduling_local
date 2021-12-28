from django.shortcuts import render
from django.db.models import Max
from django.db.models.aggregates import Count
from django.contrib.auth.models import User, Group
from django.http import JsonResponse
from django.http.response import HttpResponse
from .sslstm import sslstm
from company.models import *
from order.models import *
import pandas as pd
import json, csv, os, datetime, calendar, random
from datetime import timedelta
from calendar import monthrange
from bson import json_util
from random import randint

# 스케쥴링실행 Execute Schedule HTML
def home(request):
    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '스케쥴링 / 스케쥴링실행'
    }
    return render(request, 'schedule/schedule.html', date)

# 스케쥴링이력 Schedule history HTML
def history(request):
    date = datetime.datetime.today() - timedelta(days=3)
    if 'dateFrom' in request.GET:
        sch_date_from = datetime.datetime.strptime(request.GET['dateFrom'], "%Y-%m-%d")
        sch_date_to = datetime.datetime.strptime(request.GET['dateTo'], "%Y-%m-%d")

        date_from = request.GET['dateFrom'].replace('-', '')
        date_to = request.GET['dateTo'].replace('-', '')
        schedule_list = Schedule.objects.raw(
            "SELECT *, SUBSTRING(sch_id, 4, 8) as exe_date, SUBSTRING_INDEX(order_id, '.', 1) AS orderid, max(COUNT) as max_count " +
            "FROM company_schedule " +
            "WHERE SUBSTRING(sch_id, 4, 8) >= '" + date_from + "' AND "
                                                                             "SUBSTRING(sch_id, 4, 8) <= '" + date_to + "'" +
        "GROUP BY SUBSTRING(sch_id, 4, 8), SUBSTRING(order_id, 4, 3)")
    else:
        sch_date_from = date
        sch_date_to = datetime.datetime.today()
        schedule_list = Schedule.objects.raw(
            "SELECT *, SUBSTRING(sch_id, 4, 8) as exe_date, SUBSTRING_INDEX(order_id, '.', 1) AS orderid, max(COUNT) as max_count " +
            "FROM company_schedule " +
            "WHERE SUBSTRING(sch_id, 4, 8) >= '" + date.strftime("%Y%m%d") + "' AND "
            "SUBSTRING(sch_id, 4, 8) <= '" + datetime.datetime.today().strftime( "%Y%m%d") + "'" +
            "GROUP BY SUBSTRING(sch_id, 4, 8), SUBSTRING(order_id, 4, 3)")


    sch_list = []
    for sch in schedule_list:
        sch_dict = {}
        sch_dict['orderid'] = sch.orderid
        sch_dict['max_count'] = sch.max_count
        exe_date = datetime.datetime.strptime(sch.exe_date, "%Y%m%d")
        sch_dict['exe_date'] = exe_date.strftime("%Y-%m-%d")
        sch_date = OrderList.objects.filter(order_id=sch.orderid)
        for ord in sch_date:
            sch_date = datetime.datetime.strptime(ord.sch_date, "%Y%m%d")
            exp_date = datetime.datetime.strptime(ord.exp_date, "%Y%m%d")
            sch_dict['sch_date'] = sch_date.strftime("%Y-%m-%d")
            sch_dict['exp_date'] = exp_date.strftime("%Y-%m-%d")
        sch_list.append(sch_dict)

    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        'dateFrom': sch_date_from.strftime("%Y-%m-%d"),
        'dateTo': sch_date_to.strftime("%Y-%m-%d"),
        'path': '스케쥴링 / 스케쥴링이력',
        'sch_list' : sch_list
    }
    return render(request, 'schedule/sch_history.html', date)

# 스케쥴링이력 간트차트 Schedule history in Gantt Chart HTML
def history_chart(request, id):
    schedule_list = Schedule.objects.raw("SELECT *, SUBSTRING_INDEX(order_id, '.', 1) AS orderid, max(COUNT) as max_count " +
                                            "FROM company_schedule " +
                                            "GROUP BY SUBSTRING(sch_id, 4, 8), SUBSTRING(order_id, 4, 3)")
    sch_list = []
    for sch in schedule_list:
        sch_dict = {}
        sch_dict['orderid'] = sch.orderid
        sch_dict['max_count'] = sch.max_count
        sch_date = OrderList.objects.filter(order_id=sch.orderid)


        sch_list.append(sch_dict)

    date = datetime.datetime.today() - timedelta(days=3)
    date = {
        'dateFrom': date.strftime("%Y-%m-%d"),
        'path': '스케쥴링 / 스케쥴링이력',
        'sch_list' : sch_list
    }
    return render(request, 'schedule/sch_history_chart.html', date)

# 확정 스케쥴 Fixed Schedule HTML
def sch_confirmed(request):
    date = datetime.datetime.today()
    date = date.strftime("%Y%m%d")
    year = date[0:4]
    month = date[4:6]

    lastday = calendar.monthrange(int(year), int(month))[1]
    final_result = Schedule.objects.filter(work_end_date__gte=date)
    available_list = []
    for i in final_result:
        available_dict = {}
        available_dict['work_str_date'] = i.work_str_date
        available_dict['work_end_date'] = i.work_end_date

        querySet = OrderSchedule.objects.filter(sch_id=i.sch_id)
        if len(querySet.values()) > 0:
            for j in querySet:
                available_dict['order_id'] = j.order_id_id
                orderlist = OrderList.objects.get(order_id=j.order_id_id)
                available_dict['sch_date'] = orderlist.sch_date
                available_dict['sch_id'] = j.sch_id_id
                available_list.append(available_dict)

    date = {
        'order_list': available_list,
        'path': '스케쥴링 / 확정스케쥴조회'
    }
    return render(request, 'schedule/sch_confirmed.html', date)

# 수락한 오더 리스트 조회 (월별)
def monthly_confirmed_order(request):
    date = datetime.datetime.today().strftime("%Y%m")
    for i in request.GET:
        request = json.loads(i)
    available_list = []
    if request['str_year'] != None:
        str_year = request['str_year']
        str_month = request['str_month']
        end_year = request['end_year']
        end_month = request['end_month']

        final_result = Schedule.objects.filter(work_str_date__gte=str(str_year) + str(str_month) + '01').filter(work_end_date__lte=str(end_year) + str(end_month) + '01')
    else:
        year = date[0:4]
        month = date[4:6]

        lastday = calendar.monthrange(int(year), int(month))[1]
        final_result = Schedule.objects.filter(work_str_date__gte=date + '01').filter(
            work_end_date__lte=date + str(lastday))

    for i in final_result:
        available_dict = {}
        available_dict['work_str_date'] = i.work_str_date
        available_dict['work_end_date'] = i.work_end_date

        querySet = OrderSchedule.objects.filter(sch_id=i.sch_id)
        if len(querySet.values()) > 0:
            for j in querySet:
                available_dict['order_id'] = j.order_id_id
                available_dict['sch_id'] = j.sch_id_id
                available_list.append(available_dict)

    def json_default(value):
        if isinstance(value, datetime.datetime):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')
    # return JsonResponse(list(available_list), safe=False)
    return HttpResponse(json.dumps(available_list, default=json_default, ensure_ascii=False), content_type="application/json")

# draw Gantt Chart data
def draw_graph(request):
    count = Schedule.objects.filter(comp_id=request.user.groups.values('id')[0]['id'])
    count = count.aggregate(Max('count'))
    for i in Group.objects.all():
        for j in Group.objects.filter(name=i).values():
            schedule_list = Schedule.objects.filter(count=count['count__max'])
            schedule_list = list(schedule_list.values())
    def json_default(value):
        if isinstance(value, datetime.datetime):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')
    # print(json.dumps(schedule_list, default=json_default))
    return HttpResponse(json.dumps(schedule_list, default=json_default))

# x, y 축 저장 Save x, y axis into Database
def update_graph(request):

    # 회사명, 취급 섬유 유형 종류, 오더한 사람이 요청한 섬유 유형 종류
    orders = OrderList.objects.filter(sch_date=datetime.datetime.today().strftime("%Y%m%d"))
    textile_dict = {}
    order_list = []
    comp_list = Facility.objects.all()
    comp_ids = []
    for ord in orders:
        order_list.append(ord.order_id)
        textile_dict[ord.order_id] = ord.textile_type

    count = Schedule.objects.all()
    count = count.aggregate(Max('count'))
    count = str(count['count__max'])
    if count == 'None':
        count = '0'

    for i in textile_dict.keys():
        for comp in comp_list:
            print(comp.textile_type)
            print(textile_dict[i])
            if comp.textile_type.find(textile_dict[i]) != -1:
                comp_ids.append(comp.comp_id)

        random.shuffle(comp_ids)
        random_comp_list = comp_ids[0]
        result = {}
        result = sslstm(request) # ss-lstm 실행
        keys_list = []
        for i in result.keys():
            keys_list.append(i)

        list = []
        for i in range(len(result['id'])):
            line = []
            for keys in keys_list:
                line.append(result[keys][i])
            list.append(line)
        for i in range(len(list)):
            order_number = str(list[i][6]).split('.')
            for j in range(len(list[i])):
                    Schedule.objects.update_or_create(
                        sch_id='SCH' + list[i][0] + str(list[i][1])[2:3] + count,
                        comp_id=Information.objects.get(comp_id=request.user.groups.values('id')[0]['id']),
                        count=int(count) + 1,
                        sch_color=list[i][1],
                        order_id=str(order_list[int(order_number[0]) - 1]) + '.' + order_number[1],  # 추후 오더 데이터에서 가져오기
                        x_axis_1=list[i][2],
                        x_axis_2=list[i][3],
                        y_axis_1=list[i][4],
                        y_axis_2=list[i][5]
                    )
        break

    return JsonResponse({"message": 'success'})

# 처리 가능 오더 리스트 표출
def available_comp(request):
    # template_name = 'main/index.html'
    # available_list = Schedule.objects.filter(x_axis_1__lt='987')
    # available_list = list(available_list.values('sch_id')) # group by 맨 뒷자리, 갯수로 나눠서 3개만 나오게하기
    result = available_list(request)

    def json_default(value):
        if isinstance(value, datetime.datetime):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')
    # print(json.dumps(schedule_list, default=json_default))
    return HttpResponse(json.dumps(result, default=json_default))

# 처리 가능 오더 리스트 추출
def available_list(request):
    color_list = []
    final_list = []
    dict_list = []
    result = []

    if len(fixed_list(request))!=0:
        for i in fixed_list(request):
            if i['use_yn'] == 'Y':
                result.append(i)
            else:
                count = Schedule.objects.filter(comp_id=request.user.groups.values('id')[0]['id'])
                count = count.aggregate(Max('count'))
                count = str(count['count__max'])
                available_list = Schedule.objects.raw(
                    'SELECT sch_color, sch_id FROM company_schedule WHERE x_axis_2 < 1040 AND count = ' +
                    count + ' AND comp_id=' + str(
                        request.user.groups.values('id')[0]['id']) + ' GROUP BY sch_color HAVING COUNT(sch_color)=10')

                color_list = []
                final_list = []
                dict_list = []
                result = []
                for p in available_list:
                    color_list.append(p.sch_color)

                for i in color_list:
                    # color = Schedule.objects.raw('SELECT * FROM company_schedule WHERE sch_color = "' +  i + '" GROUP BY sch_color HAVING MAX(sch_id)')
                    color = Schedule.objects.filter(sch_color=i, comp_id=request.user.groups.values('id')[0]['id'])
                    final_list.append(color.aggregate(Max('sch_id')))

                for i in range(len(final_list)):
                    dict_list.append(final_list[i]['sch_id__max'])

                for i in dict_list:
                    schedule_list = Schedule.objects.filter(sch_id=i)
                    result.append(list(schedule_list.values()))
    else:
        count = Schedule.objects.filter(comp_id=request.user.groups.values('id')[0]['id'])
        count = count.aggregate(Max('count'))
        count = str(count['count__max'])
        available_list = Schedule.objects.raw(
            'SELECT sch_color, sch_id FROM company_schedule WHERE x_axis_2 < 1040 AND count = ' +
            count + ' AND comp_id=' + str(request.user.groups.values('id')[0]['id']) + ' GROUP BY sch_color HAVING COUNT(sch_color)=10')


        for p in available_list:
            color_list.append(p.sch_color)

        for i in color_list:
            # color = Schedule.objects.raw('SELECT * FROM company_schedule WHERE sch_color = "' +  i + '" GROUP BY sch_color HAVING MAX(sch_id)')
            color = Schedule.objects.filter(sch_color=i, comp_id=request.user.groups.values('id')[0]['id'])
            final_list.append(color.aggregate(Max('sch_id')))

        for i in range(len(final_list)):
            dict_list.append(final_list[i]['sch_id__max'])

        for i in dict_list:
            schedule_list = Schedule.objects.filter(sch_id=i)
            result.append(list(schedule_list.values()))

    return result

# 처리 확정 히스토리 표출
def schedule_history(request):
    result = fixed_list(request)

    def json_default(value):
        if isinstance(value, datetime.datetime):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')

    # print(json.dumps(schedule_list, default=json_default))
    return HttpResponse(json.dumps(result, default=json_default))

# 처리 확정 히스토리 리스트 추출
def fixed_list(request):
    date = datetime.datetime.today().strftime("%Y%m%d")
    available_list = OrderSchedule.objects.raw(
        'SELECT * FROM order_orderschedule WHERE SUBSTRING(sch_id, 4, 8) = ' + date)
    result = []
    for p in available_list:
        avail_dict = {}
        avail_dict['use_yn'] = p.use_yn
        avail_dict['schedule'] = list(
            Schedule.objects.filter(comp_id=request.user.groups.values('id')[0]['id'], sch_id=p.sch_id.sch_id).values())

        if len(avail_dict['schedule']) == 0:
            continue;
        else:
            result.append(avail_dict)

    return result

# 처리 가능 오더 확정
def fixed_order(request):
    orders = request.POST.getlist('order_list[]')
    if len(fixed_list(request)) > 0:
        result = fixed_list(request)
        for sch in result:
            OrderSchedule.objects.filter(sch_id=sch).update(use_yn='N')

    # 오더 아이디로 넘어옴
    for ord in orders:
        orders = Schedule.objects.filter(sch_id=ord)  # order_id 나중에 order_list 에서 조회해 오기
        if not orders:
            OrderSchedule.objects.filter(sch_id=ord).update(use_yn='Y')
        else:
            for i in orders:
                OrderSchedule.objects.update_or_create(
                    # order_id=OrderList.objects.get(order_id=ord.split('.')[0]),
                    order_id=OrderList.objects.get(order_id=i.order_id.split('.')[0]),
                    sch_id=Schedule.objects.get(sch_id=i.sch_id),
                    use_yn='Y'
                )

    return JsonResponse({"message": 'success'})

# 수락한 오더 리스트 조회
def confirmed_order(request):
    result = available_list(request)
    final_result = []
    for i in range(len(result)):
        try:
            sch_id = result[i]['schedule'][0]['sch_id']
            confirmed = OrderSchedule.objects.filter(sch_id=sch_id).values()
        except:
            sch_id = result[i][0]['sch_id']
            confirmed = OrderSchedule.objects.filter(sch_id=sch_id).values()

    final_result=list(confirmed)
    def json_default(value):
        if isinstance(value, datetime.datetime):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')

    # print(json.dumps(schedule_list, default=json_default))
    return HttpResponse(json.dumps(final_result, default=json_default))


# 스케줄 새로 생성시 삭제
def delete_order(request):
    # orders = request.POST.getlist('order_list[]')
    schs = request.POST.getlist('sch_list[]')
    # 오더 아이디로 넘어옴
    for sch in schs:
        OrderSchedule.objects.filter(sch_id=sch).update(use_yn = 'N')

        # OrderSchedule.objects.update_or_create(
            # use_yn='N',
        # )
        # OrderSchedule.objects.filter(order_id=ord.split('.')[0]).delete()  # order_id 나중에 order_list 에서 조회해 오기

    return JsonResponse({"message": 'success'})


def test_data(request):
    date = datetime.datetime.today().strftime("%Y%m")
    for i in request.GET:
        request = json.loads(i)
    available_list = []
    year = date[0:4]
    month = date[4:6]
    lastday = calendar.monthrange(int(year), int(month))[1]
    final_result = Schedule.objects.filter(work_str_date__gte=date + '01').filter(
        work_end_date__lte=date + str(lastday))
    for i in final_result:
        available_list.append(OrderSchedule.objects.filter(sch_id=i.sch_id))
    def json_default(value):
        if isinstance(value, datetime.datetime):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')

    # print(json.dumps(schedule_list, default=json_default))
    return HttpResponse(json.dumps(list(final_result.values()), default=json_default))
