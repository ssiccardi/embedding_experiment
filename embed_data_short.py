# reads events data written by preprocess_data.py and builds an "embedding" with events that happened 10 minutes
#   before and 10 minutes after at the same atm. Embedding columns are:
#   - first 4 cols contain number of type A, B, M, N events; 
#   - next 4 cols contain 1 if the event is of type A, 0 otherwise; the same for types B, M, N
#   - next 2 cols contain number of sutypes of A events and B events found
#   - next 2 cols contain number of A events with same subtype as the event, and the same for B events

# parameters: folder where the files can be found, N of minutes to search for events, Days to process
# input files: typed_events_0n.csv;
# output file: embedded_events.csv

from optparse import OptionParser
import csv
from datetime import datetime
#import pandas as pd


parser = OptionParser()
parser.add_option("-F", dest="in_folder",
                  help="folder of input csv")
parser.add_option("-N", dest="minutes",
                  help="max n. of minutes to scan")
parser.add_option("-D", dest="dayproc",
                  help="days to process (<10)")
(optlist, args) = parser.parse_args()

assert optlist.in_folder, "The input folder is mandatory (run using -F file.csv)"
assert optlist.dayproc, "The days to process is mandatory (run using -D n)"

dayproc = int(optlist.dayproc)
in_folder = optlist.in_folder
if in_folder[-1] != "/":
    in_folder = in_folder + "/"

if optlist.minutes:
    minutes = int(optlist.minutes)*60
else:
    minutes = 600 # default

print(datetime.now())

ofile = open(in_folder+ "embedded_events.csv","w")
for i in range(12):
    ofile.write('"e'+str(i)+'",')
ofile.write('"label"\r\n')

tmpini = [0,0,0,0,0,0,0,0,0,0,0,0]
print(len(tmpini))
scritti = 0
for i in range(1,dayproc):
    source = open(in_folder+ "typed_events_0"+str(i)+".csv")
    reader = csv.reader(source, delimiter=",")
    n_wit=1  # to compute short code for debit cards
    for rec in reader:
        if not rec[0]:
            continue
        if rec[0] == "atm":
            savatm = "atm"
            tmparr = []
            tmpta = []
            tmptb = []
            continue
        if rec[0] != savatm:
        # generation of embedding
            for k in range(len(tmparr)):
                tmp = tmpini.copy()
                tmpta = []
                tmptb = []
                ev = tmparr[k]
                tmp[ev[0]+3] = 1   # 1 in position 5 (index 4) to 8 (index 7) depending on event type
                basetime = ev[1]
                for j in range(k-1,0,-1):
                # go back to find previous events within the max search time, if any
                    if basetime - tmparr[j][1] > minutes:
                        break
                    tmp[tmparr[j][0]-1] = tmp[tmparr[j][0]-1] + 1 # add 1 in position 1 (index 0) to 4 (index 3) depending on event type
                    if tmparr[j][0] == 1:
                    # count subtypes of A events
                        if not tmparr[j][3] in tmpta:
                        # new subtype
                            tmp[8] = tmp[8] + 1
                            tmpta.append(tmparr[j][3])
                        if ev[0] == 1 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[10] = tmp[10] + 1
                    elif tmparr[j][0] == 2:
                    # count subtypes of B events
                        if not tmparr[j][3] in tmptb:
                        # new subtype
                            tmp[9] = tmp[9] + 1
                            tmptb.append(tmparr[j][3])
                        if ev[0] == 2 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[11] = tmp[11] + 1
                for j in range(k+1,len(tmparr)):
                # go forward to find previous events within the max search time, if any
                    if tmparr[j][1] - basetime > minutes:
                        break
                    tmp[tmparr[j][0]-1] = tmp[tmparr[j][0]-1] + 1 # add 1 in position 1 (index 0) to 4 (index 3) depending on event type
                    if tmparr[j][0] == 1:
                    # count subtypes of A events
                        if not tmparr[j][3] in tmpta:
                        # new subtype
                            tmp[8] = tmp[8] + 1
                            tmpta.append(tmparr[j][3])
                        if ev[0] == 1 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[10] = tmp[10] + 1
                    elif tmparr[j][0] == 2:
                    # count subtypes of B events
                        if not tmparr[j][3] in tmptb:
                        # new subtype
                            tmp[9] = tmp[9] + 1
                            tmptb.append(tmparr[j][3])
                        if ev[0] == 2 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[11] = tmp[11] + 1
                if ev[2] == '1':
                    nwrites = 20
                else:
                    nwrites = 1
                # I want the number of criminal events about 10% of total, so I write each 20 times
                for l in range(nwrites):
                    for j in range(len(tmp)):
                        ofile.write(str(tmp[j])+',')
                    ofile.write(str(ev[2])+'\r\n')
                    scritti = scritti + 1
            savatm = rec[0]
            tmparr = []
        # prepare a list of events of the same atm
        if rec[7][:1] == "A":
            tmpt = 1
            tmpt1 = int(rec[7][1:])
        elif rec[7][:1] == "B":
            tmpt = 2
            tmpt1 = int(rec[7][1:])
        elif rec[7][:1] == "M":
            tmpt = 3
            tmpt1 = 0
        else:
            tmpt = 4
            tmpt1 = 0
        if rec[8] == 'n':
            tmpl = "0"
        else:
            tmpl = "1"
        tmparr.append([tmpt,int(rec[2]),tmpl,tmpt1])  # converted type and time; label; subtype for A and B events
        n_wit=n_wit+1
    for k in range(len(tmparr)):
                tmp = tmpini.copy()
                tmpta = []
                tmptb = []
                ev = tmparr[k]
                tmp[ev[0]+3] = 1   # 1 in position 5 (index 4) to 8 (index 7) depending on event type
                basetime = ev[1]
                for j in range(k-1,0,-1):
                # go back to find previous events within the max search time, if any
                    if basetime - tmparr[j][1] > minutes:
                        break
                    tmp[tmparr[j][0]-1] = tmp[tmparr[j][0]-1] + 1 # add 1 in position 1 (index 0) to 4 (index 3) depending on event type
                    if tmparr[j][0] == 1:
                    # count subtypes of A events
                        if not tmparr[j][3] in tmpta:
                        # new subtype
                            tmp[8] = tmp[8] + 1
                            tmpta.append(tmparr[j][3])
                        if ev[0] == 1 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[10] = tmp[10] + 1
                    elif tmparr[j][0] == 2:
                    # count subtypes of B events
                        if not tmparr[j][3] in tmptb:
                        # new subtype
                            tmp[9] = tmp[9] + 1
                            tmptb.append(tmparr[j][3])
                        if ev[0] == 2 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[11] = tmp[11] + 1
                for j in range(k+1,len(tmparr)):
                # go forward to find previous events within the max search time, if any
                    if tmparr[j][1] - basetime > minutes:
                        break
                    tmp[tmparr[j][0]-1] = tmp[tmparr[j][0]-1] + 1 # add 1 in position 1 (index 0) to 4 (index 3) depending on event type
                    if tmparr[j][0] == 1:
                    # count subtypes of A events
                        if not tmparr[j][3] in tmpta:
                        # new subtype
                            tmp[8] = tmp[8] + 1
                            tmpta.append(tmparr[j][3])
                        if ev[0] == 1 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[10] = tmp[10] + 1
                    elif tmparr[j][0] == 2:
                    # count subtypes of B events
                        if not tmparr[j][3] in tmptb:
                        # new subtype
                            tmp[9] = tmp[9] + 1
                            tmptb.append(tmparr[j][3])
                        if ev[0] == 2 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[11] = tmp[11] + 1
                if ev[2] == '1':
                    nwrites = 20
                else:
                    nwrites = 1
                # I want the number of criminal events about 10% of total, so I write each 20 times
                for l in range(nwrites):
                    for j in range(len(tmp)):
                        ofile.write(str(tmp[j])+',')
                    ofile.write(str(ev[2])+'\r\n')
                    scritti = scritti + 1


    source.close()
    print("Events read: "+str(n_wit)+ " written "+str(scritti))


ofile.close()

ofile = open(in_folder+ "embedded_events_test.csv","w")
for i in range(12):
    ofile.write('"e'+str(i)+'",')
ofile.write('"label"\r\n')

for i in range(dayproc,dayproc+1):
    source = open(in_folder+ "typed_events_0"+str(i)+".csv")
    reader = csv.reader(source, delimiter=",")
    n_wit=1  # to compute short code for debit cards

    for rec in reader:
        if not rec[0]:
            continue
        if rec[0] == "atm":
            savatm = "atm"
            tmparr = []
            tmpta = []
            tmptb = []
            continue
        if rec[0] != savatm:
        # generation of embedding
            for k in range(len(tmparr)):
                tmp = tmpini.copy()
                tmpta = []
                tmptb = []
                ev = tmparr[k]
                tmp[ev[0]+3] = 1   # 1 in position 5 (index 4) to 8 (index 7) depending on event type
                basetime = ev[1]
                for j in range(k-1,0,-1):
                # go back to find previous events within the max search time, if any
                    if basetime - tmparr[j][1] > minutes:
                        break
                    tmp[tmparr[j][0]-1] = tmp[tmparr[j][0]-1] + 1 # add 1 in position 1 (index 0) to 4 (index 3) depending on event type
                    if tmparr[j][0] == 1:
                    # count subtypes of A events
                        if not tmparr[j][3] in tmpta:
                        # new subtype
                            tmp[8] = tmp[8] + 1
                            tmpta.append(tmparr[j][3])
                        if ev[0] == 1 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[10] = tmp[10] + 1
                    elif tmparr[j][0] == 2:
                    # count subtypes of B events
                        if not tmparr[j][3] in tmptb:
                        # new subtype
                            tmp[9] = tmp[9] + 1
                            tmptb.append(tmparr[j][3])
                        if ev[0] == 2 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[11] = tmp[11] + 1
                for j in range(k+1,len(tmparr)):
                # go forward to find previous events within the max search time, if any
                    if tmparr[j][1] - basetime > minutes:
                        break
                    tmp[tmparr[j][0]-1] = tmp[tmparr[j][0]-1] + 1 # add 1 in position 1 (index 0) to 4 (index 3) depending on event type
                    if tmparr[j][0] == 1:
                    # count subtypes of A events
                        if not tmparr[j][3] in tmpta:
                        # new subtype
                            tmp[8] = tmp[8] + 1
                            tmpta.append(tmparr[j][3])
                        if ev[0] == 1 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[10] = tmp[10] + 1
                    elif tmparr[j][0] == 2:
                    # count subtypes of B events
                        if not tmparr[j][3] in tmptb:
                        # new subtype
                            tmp[9] = tmp[9] + 1
                            tmptb.append(tmparr[j][3])
                        if ev[0] == 2 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[11] = tmp[11] + 1
#                if ev[2] == '1':
#                    nwrites = 20
#                else:
                nwrites = 1
                # I want the number of criminal events about 10% of total, so I write each 20 times
                for l in range(nwrites):
                    for j in range(len(tmp)):
                        ofile.write(str(tmp[j])+',')
                    ofile.write(str(ev[2])+'\r\n')
                    scritti = scritti + 1
            savatm = rec[0]
            tmparr = []
        # prepare a list of events of the same atm
        if rec[7][:1] == "A":
            tmpt = 1
            tmpt1 = int(rec[7][1:])
        elif rec[7][:1] == "B":
            tmpt = 2
            tmpt1 = int(rec[7][1:])
        elif rec[7][:1] == "M":
            tmpt = 3
            tmpt1 = 0
        else:
            tmpt = 4
            tmpt1 = 0
        if rec[8] == 'n':
            tmpl = "0"
        else:
            tmpl = "1"
        tmparr.append([tmpt,int(rec[2]),tmpl,tmpt1])  # converted type and time; label; subtype for A and B events
        n_wit=n_wit+1
    for k in range(len(tmparr)):
                tmp = tmpini.copy()
                tmpta = []
                tmptb = []
                ev = tmparr[k]
                tmp[ev[0]+3] = 1   # 1 in position 5 (index 4) to 8 (index 7) depending on event type
                basetime = ev[1]
                for j in range(k-1,0,-1):
                # go back to find previous events within the max search time, if any
                    if basetime - tmparr[j][1] > minutes:
                        break
                    tmp[tmparr[j][0]-1] = tmp[tmparr[j][0]-1] + 1 # add 1 in position 1 (index 0) to 4 (index 3) depending on event type
                    if tmparr[j][0] == 1:
                    # count subtypes of A events
                        if not tmparr[j][3] in tmpta:
                        # new subtype
                            tmp[8] = tmp[8] + 1
                            tmpta.append(tmparr[j][3])
                        if ev[0] == 1 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[10] = tmp[10] + 1
                    elif tmparr[j][0] == 2:
                    # count subtypes of B events
                        if not tmparr[j][3] in tmptb:
                        # new subtype
                            tmp[9] = tmp[9] + 1
                            tmptb.append(tmparr[j][3])
                        if ev[0] == 2 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[11] = tmp[11] + 1
                for j in range(k+1,len(tmparr)):
                # go forward to find previous events within the max search time, if any
                    if tmparr[j][1] - basetime > minutes:
                        break
                    tmp[tmparr[j][0]-1] = tmp[tmparr[j][0]-1] + 1 # add 1 in position 1 (index 0) to 4 (index 3) depending on event type
                    if tmparr[j][0] == 1:
                    # count subtypes of A events
                        if not tmparr[j][3] in tmpta:
                        # new subtype
                            tmp[8] = tmp[8] + 1
                            tmpta.append(tmparr[j][3])
                        if ev[0] == 1 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[10] = tmp[10] + 1
                    elif tmparr[j][0] == 2:
                    # count subtypes of B events
                        if not tmparr[j][3] in tmptb:
                        # new subtype
                            tmp[9] = tmp[9] + 1
                            tmptb.append(tmparr[j][3])
                        if ev[0] == 2 and ev[3] == tmparr[j][3]:
                        # same subtype as event, add in position 11 (index 10)
                            tmp[11] = tmp[11] + 1
#                if ev[2] == '1':
#                    nwrites = 20
#                else:
                nwrites = 1
                # I want the number of criminal events about 10% of total, so I write each 20 times
                for l in range(nwrites):
                    for j in range(len(tmp)):
                        ofile.write(str(tmp[j])+',')
                    ofile.write(str(ev[2])+'\r\n')
                    scritti = scritti + 1

    source.close()
    print("Events read: "+str(n_wit))


ofile.close()

print(datetime.now())
