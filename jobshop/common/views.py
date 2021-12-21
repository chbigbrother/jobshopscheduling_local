from django.shortcuts import render

# Create your views here.
def id_generate(request, id):
    if id is None:
        int_id = 0
    else:
        int_id = id
    int_id = int(id) + 1
    str_id = request

    if int_id < 10:
        str_id = 'ORD00' + str(int_id)
    if int_id > 10:
        str_id = 'ORD0' + str(int_id)
    if int_id > 99:
        str_id = 'ORD' + str(int_id)
    print(str_id)
    return str_id