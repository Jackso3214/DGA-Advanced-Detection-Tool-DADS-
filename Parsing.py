
import csv
import re
import argparse

parser = argparse.ArgumentParser(description='Add a filename')
parser.add_argument('-i', '--input', help='Input a filename', required=True) 
args = parser.parse_args()
file = open("D:\Projects\Python\LogisticRegressionPractice\dns.txt", encoding='utf-16-le')
num=[]
host=[]
port=[]
request_time=[]
response_time=[]
src_address=[]
dest_address=[]
x=0
field_names=['id','hostname','port number','request time','response time','source address','destination address']
for line in file:
    if ':' in line:
        line_values = line.split(':')
        label=line_values[0]
        value=line_values[1]
        num.append(x)
        if label=="Host" in label:
            host.append(value)
            x=x+1
        if label=="Port" in label:
            
            port.append(value)
            
        if label=="Request" in label:
            
            request_time.append(value)
            
        if label=="Response" in label:
            
            response_time.append(value)
            
        if label=="Source" in label:
            
            src_address.append(value)
            
        if label=="Destination" in label:
            
            dest_address.append(value)
            
outputfile = open('output.csv', 'w')
writer=csv.writer(outputfile)
writer.writerow(field_names)
for y in num:
    currentline=[num[y],host[y],port[y],request_time[y],response_time[y],src_address[y],dest_address[y]]
writer.close    
    







