# reads data for atm (with nearest tower), bankoffices (with phones), debit cards (with banks)
#   and withdrawals (normal+criminal 1 day at a time) and builds a table with:
#   atm, near tower, time, amount, debit card phone, bank phone
#   ordered by atm and time
#   then reads phone calls data (normal+criminal 1 day at a time) and adds to the table the
#   following:
#   - tower receiving the first call from the bank phone to the card phone in N minutes after the time
#     of withdrawal (N = parameter, None if no such call)
#   - withdrawal type (Ax = max amount, different towers, where x is the receiving tower; N = same tower, any amounts; 
#     Bx = non max amout, different towers, where x is the receiving tower; M = no call found, any amounts)
#   writes to file: atm, time, amount, debit card phone, far tower, type, normal or criminal (to train and test the models)
#   in order of atm, time
# We want to find patterns of types within a time interval from the first and for the same atm, that are similar to
#   a predefined pattern.
# In a second run we will check if the involved phones belong to the same set for all the similar patterns found

# parameters: folder where the files can be found, N of minutes for the call from the bank, Day to process
# input files: atm_cell_tower_rels.csv, bank_offices.csv, debit_cards.csv;
#              generated/users_01_withdrawals.csv and generated/co_01_withdrawals.csv (01-05)
#              generated/users_01_phone_calls.csv and generated/co_01_phone_calls.csv (01-05)
# output file: typed_events.csv

from optparse import OptionParser
import csv
from datetime import datetime
import pandas as pd


parser = OptionParser()
parser.add_option("-F", dest="in_folder",
                  help="folder of input csv")
parser.add_option("-N", dest="minutes",
                  help="max n. of minutes to receive sms after withdrawal")
parser.add_option("-D", dest="dayproc",
                  help="day to process (1-5)")
(optlist, args) = parser.parse_args()

assert optlist.in_folder, "The input folder is mandatory (run using -F file.csv)"
assert optlist.dayproc, "The day to process is mandatory (run using -D n)"

dayproc = optlist.dayproc
in_folder = optlist.in_folder
if in_folder[-1] != "/":
    in_folder = in_folder + "/"

if optlist.minutes:
    minutes = int(optlist.minutes)*60
else:
    minutes = 600 # default

print(datetime.now())

towers = {}
source = open(in_folder+ "torri_tim_vodafone.csv")
reader = csv.reader(source, delimiter=",")
n_tow=1  # to compute short code for tower
for rec in reader:
#    print(rec)
    if rec[0] == "id":
        continue
    if not float(rec[0]) in towers:
        towers[float(rec[0])] = "t"+str(n_tow)
        #print(str(float(rec[0])))
        n_tow = n_tow + 1
print("Towers read: "+str(n_tow-1), len(towers))
print(towers[29175.1204])
source.close()
# now given a tower long code I can find tower short code


atms = {}
source = open(in_folder+ "atm_cell_tower_rels.csv")
reader = csv.reader(source, delimiter=",")
n_atm=1  # to compute short code for atm
n_tow=1  # to compute short code for tower
for rec in reader:
#    print(rec)
    if rec[0] == "id":
        continue
    if not float(rec[2]) in towers:
        towers[float(rec[2])] = "t"+str(n_tow)
        n_tow = n_tow + 1
    towcode = towers[float(rec[2])]
    atms[rec[1]] = ["a"+str(n_atm), towcode]
    n_atm = n_atm+1
print("ATMs read: "+str(n_atm-1)+", connected towers added: "+str(n_tow-1))
source.close()
# now given an atm long code I can find atm short code and nearest tower short code

source = open(in_folder+ "bank_offices.csv")
reader = csv.reader(source, delimiter=",")
banks = {}
phones = {}
n_ban=1  # to count banks
n_pho=1  # to compute short code for phones
for rec in reader:
    if not rec[0]:
        continue
    if rec[0] == "id":
        continue
    if not rec[5] in phones:
        phones[rec[5]] = "p"+str(n_pho)
        n_pho = n_pho + 1
    phonecode = phones[rec[5]]
    banks[rec[0]] = [phonecode,rec[5]]  # bank phone number
    n_ban = n_ban+1
print("Banks read: "+str(n_ban-1)+", phones read: "+str(n_pho-1))
source.close()
# now given a bank I can find its phone number and short code (in 1 step)


source = open(in_folder+ "debit_cards.csv")
reader = csv.reader(source, delimiter=",")
debit_cards = {}
n_dca=1  # to compute short code for debit cards
i = 1
for rec in reader:
    if not rec[0]:
        continue
    if rec[0] == "id":
        continue
    if not rec[2] in phones:
        phones[rec[2]] = "p"+str(n_pho)
        n_pho = n_pho + 1
    phonecode = phones[rec[2]]
    debit_cards[rec[0]] = [banks[rec[1]][0], phonecode, banks[rec[1]][1], rec[2]]  # bank phone code, phone code of the card; bank phone number, card phone number
    n_dca = n_dca+1
print("Cards read: "+str(n_dca-1)+", phones read: "+str(n_pho-1))
source.close()
# now given a debit card I can find its phone short code and its bank's phone short code

source = open(in_folder+ "generated/users_0"+dayproc+"_withdrawals.csv")
reader = csv.reader(source, delimiter=",")
withdrawals = []
n_wit=1  # to compute short code for debit cards
i = 1
#   atm, near tower, time, amount, debit card phone, bank phone
for rec in reader:
    if not rec[0]:
        continue
    if rec[0] == "id":
        continue
    atmcode = atms[rec[2]][0]
    towcode = atms[rec[2]][1]
    wtime = int(rec[1])
    amount = int(rec[4])
    dcphonecod = debit_cards[rec[3]][1]
    bkphonecod = debit_cards[rec[3]][0]
    dcphonenum = debit_cards[rec[3]][3]
    bkphonenum = debit_cards[rec[3]][2]
    withdrawals.append([atmcode, towcode, wtime, amount, dcphonecod, bkphonecod, "", "", "n",dcphonenum, bkphonenum]) 
    # code of the tower receiving the sms, withdrawal type, criminal flag still to be computed
    n_wit=n_wit+1

source.close()
print("Normal withdrawals read: "+str(n_wit))

source = open(in_folder+ "generated/co_0"+dayproc+"_withdrawals.csv")
reader = csv.reader(source, delimiter=",")
n_wit=0  # to compute short code for debit cards
i = 1
#   atm, near tower, time, amount, debit card phone, bank phone
for rec in reader:
    if not rec[0]:
        continue
    if rec[0] == "id":
        continue
    atmcode = atms[rec[2]][0]
    towcode = atms[rec[2]][1]
    wtime = int(rec[1])
    amount = int(rec[4])
    dcphonecod = debit_cards[rec[3]][1]
    bkphonecod = debit_cards[rec[3]][0]
    dcphonenum = debit_cards[rec[3]][3]
    bkphonenum = debit_cards[rec[3]][2]
    withdrawals.append([atmcode, towcode, wtime, amount, dcphonecod, bkphonecod, "", "", "c",dcphonenum, bkphonenum]) 
    # code of the tower receiving the sms, withdrawal type, criminal flag still to be computed
    n_wit=n_wit+1

source.close()
print("Criminal withdrawals read: "+str(n_wit))

withdrawals.sort(key=lambda tup: (tup[0],tup[2]))
ofile = open(in_folder+ "generated/typed_events_0"+dayproc+".csv","w")
ofile.write('"atm","atm tower","time","amount","card phone","bank phone","recv tower","type","flag"\r\n')


pcalls = pd.concat((pd.read_csv(in_folder+"generated/users_0"+dayproc+"_phone_calls.csv", sep=","),pd.read_csv(in_folder+"generated/co_0"+dayproc+"_phone_calls.csv", sep=",")))

n_wit = 0
for w in withdrawals:
#    print(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8])
    x=pcalls.loc[(pcalls['from_number']==w[10]) & (pcalls['to_number']==w[9]) & (pcalls['time']>=w[2]) & (pcalls['time']<=w[2]+minutes)]
    if len(x) == 0:
        type = "M"
        recvt = ""
    elif len(x) == 1:        
        recvt = towers[x.iloc[0]['to_cell_tower'].round(5)]
        if recvt == w[1]:
            type = "N"
        else:
            if w[3] == 250:
                type = "A"+recvt[1:]
            else:
                type = "B"+recvt[1:]
    else:
        x=x.sort_values(by=['time'])
        recvt = towers[x.iloc[0]['to_cell_tower'].round(5)]
        if recvt == w[1]:
            type = "N"
        else:
            if w[3] == 250:
                type = "A"+recvt[1:]
            else:
                type = "B"+recvt[1:]
    ofile.write('"'+w[0]+'","'+w[1]+'",'+str(w[2])+','+str(w[3])+',"'+w[4]+'","'+w[5]+'","'+recvt+'","'+type+'","'+w[8]+'"\r\n')
    n_wit = n_wit + 1
    if n_wit % 1000 == 0:
        print(n_wit)
ofile.close()

print(datetime.now())
