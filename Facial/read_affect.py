import csv

def ReadAffect():
    file_Path = "C:\\Users\\Jiaming Nie\\Downloads\\AffectNet\\affectnet-csv\\affectnet.csv"
    csvFile = open(file_Path,"r")
    reader = csv.DictReader(csvFile)

    print(len(reader.fieldnames))

ReadAffect()