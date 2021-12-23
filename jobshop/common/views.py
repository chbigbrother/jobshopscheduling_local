from django.shortcuts import render

# Create your views here.
def id_generate(request, id):

    if id is None or not id:
        id = 0
    else:
        int_id = id

    int_id = int(id) + 1
    str_id = request

    if int_id < 10:
        str_id = request + '00' + str(int_id)
    if int_id >= 10:
        str_id = request + '0' + str(int_id)
    if int_id > 99:
        str_id = request + str(int_id)

    return str_id