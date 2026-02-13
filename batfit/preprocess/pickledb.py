import pickle
import random
import sys
import time


class PickleDB:
    def __init__(self, filename, read_from_existing=False):
        self.filename = filename
        if read_from_existing:
            self.n_data = self.get_n_data()
        else:
            self.n_data = 0

    def get_n_data(self):
        count = 0
        with open(self.filename, "rb") as f:
            while True:
                try:
                    tmp_count = pickle.load(f)
                    count += 1
                except EOFError:
                    break
        return int(count)

    def _read(self):
        records = []
        with open(self.filename, "rb") as f:
            while True:
                try:
                    records.append(pickle.load(f))
                except EOFError:
                    break
        self.n_data = len(records)
        return records

    def read(self, max_try=10):
        read_success = False
        try_count = 0
        while not read_success and try_count < max_try:
            try:
                records = self._read()
                read_success = True
            except OSError:
                try_count += 1
                time.sleep(random.random() + 1)
            except FileNotFoundError:
                sys.exit("ERROR: database must be created before it is read")
        if try_count >= max_try:
            sys.exit(f"ERROR: could not read DB after {max_try} tries")

        return records

    def _append(self, data):
        with open(self.filename, "ab+") as f:
            pickle.dump(data, f)
            self.n_data = int(max(self.n_data, data["sim_id"]))

    def append(self, data, max_try=10):
        write_success = False
        try_count = 0
        while not write_success and try_count < max_try:
            try:
                self._append(data)
                write_success = True
            except OSError:
                try_count += 1
                time.sleep(random.random() + 1)
            except FileNotFoundError:
                sys.exit(
                    "ERROR: database must be created before it is written to"
                )
        if try_count >= max_try:
            sys.exit(f"ERROR: could not write to DB after {max_try} tries")
