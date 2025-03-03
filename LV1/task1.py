work_hours = int(input("Working hours: "))
hourly_rate = float(input("Hourly rate: "))

def total_euro(work_hours, hourly_rate):
    return work_hours * hourly_rate

print('Your salary: ', total_euro(work_hours, hourly_rate))