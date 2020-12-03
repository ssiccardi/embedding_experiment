Here is a small experiment of event data embedding.

1. we have 2 types of basic events, A = withdrawals and B = phone calls. They may happen indipentendly or a withdrawal may trigger a phone call. We are interested in the latter case.
2. we can build compound events C = withdrawal + triggered phone call (if any) and assign a type to each of these events, depending on the characteristics of the withdrawal and the call
3. we are looking for a method that is able to find sequences of the compound events types, that are similar to a target sequence of types
4. we can, if needed, do some post processing (e.g. to check which debit cards are involved in the found sequences, or to find the final phone calls to the supervisor...)

Program preprocess_data.py builds the compund events files. It has:
input files:   atm_cell_tower_rels.csv, bank_offices.csv, debit_cards.csv;
              generated/users_01_withdrawals.csv and generated/co_01_withdrawals.csv (01-05)
              generated/users_01_phone_calls.csv and generated/co_01_phone_calls.csv (01-05)
output file: typed_events.csv

Program embed_data_short.py builds a very simple embedding of compunds events. It takes into account events that happened 10 minutes
   before and 10 minutes after a given event at the same atm. Embedding columns are:
   - first 4 cols contain number of type A, B, M, N events; 
   - next 4 cols contain 1 if the event is of type A, 0 otherwise; the same for types B, M, N
   - next 2 cols contain number of sutypes of A events and B events found
   - next 2 cols contain number of A events with same subtype as the event, and the same for B events
It builds 2 output files: the first contains 4 days data and is used to train and validate models, the other contains 1 day data and is used to test.
As we have only less than 1% of criminal events, they are repeated 20 times to ease the ML task.

Program example_nn_short.py is a simple neural network that usues the above files as input and tries to classify normal and criminal events.
As a 0 label is assigned to normal events and a 1 label to criminal ones, predictions are rounded and a threshold is used when comparing to targets.