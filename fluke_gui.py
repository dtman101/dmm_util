#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
PyQt5 GUI for the Fluke 28x DMM Utility.

This script combines the GUI logic with the core communication functions
from the original dmm_util.py, creating a standalone application.
"""

# --- Original dmm_util.py code (namespaced) ---
# We will define all the core logic from dmm_util.py here,
# slightly modified to be methods of a class or standalone functions
# that can be called by the GUI's worker thread.

import serial
import time
import struct
import sys
import datetime
import calendar
import binascii
from serial.tools.list_ports import comports

# Try to import PyQt5 components
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QPushButton, QComboBox, QLineEdit, QSpinBox,
        QDoubleSpinBox, QGroupBox, QTextEdit, QStatusBar, QFileDialog,
        QMessageBox, QTabWidget, QLabel, QRadioButton, QStyleFactory, QCheckBox
    )
    from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
    from PyQt5.QtCore import Qt # Import Qt
    from PyQt5.QtGui import QFont # Import QFont for monospace
except ImportError:
    print("Error: PyQt5 is not installed.")
    print("Please install it using: pip install PyQt5")
    sys.exit(1)

# --- Core DMM Logic (from dmm_util.py) ---
# We'll make these functions part of the DMMWorker or call them from it.
# Global-like variables that the original script used will be managed
# as instance attributes of the DMMWorker.

def get_u16(string, offset):
    """Unpack a 16-bit unsigned integer from byte string."""
    endian = string[offset + 1:offset - 1:-1] if offset > 0 else string[offset + 1::-1]
    return struct.unpack('!H', endian)[0]

def get_s16(string, offset):
    """Unpack a 16-bit signed integer from byte string."""
    val = get_u16(string, offset)
    if val & 0x8000 != 0:
        val = -(0x10000 - val)
    return val

def get_double(string, offset):
    """Unpack a 64-bit double from byte string."""
    endian_l = string[offset + 3:offset - 1:-1] if offset > 0 else string[offset + 3::-1]
    endian_h = string[offset + 7:offset + 3:-1]
    endian = endian_l + endian_h
    return round(struct.unpack('!d', endian)[0], 8)

def parse_time(t):
    """Convert DMM timestamp (double) to a time.struct_time."""
    try:
        return time.gmtime(t)
    except (OSError, ValueError):
        # Handle potential invalid timestamps from the device
        return time.gmtime(0)


def get_time(string, offset):
    """Get and parse a time value from a byte string."""
    return parse_time(get_double(string, offset))

def format_duration(start_time, end_time):
    """Format duration in seconds to a d:hh:mm:ss string."""
    try:
        seconds = time.mktime(end_time) - time.mktime(start_time)
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return f'{d:02d}:{h:02d}:{m:02d}:{s:02d}'
    except (OSError, OverflowError):
        return "??:??:??:??"


class DMMWorker(QObject):
    """
    Runs all DMM communication in a separate thread to keep the GUI responsive.
    """
    # Signal to send text output to the main window's text area
    output = pyqtSignal(str)
    
    # Signal to report an error as a string
    error = pyqtSignal(str)
    
    # Signal to report success/completion
    finished = pyqtSignal()
    
    # Signal to update live view data
    live_data = pyqtSignal(str, str, str) # value, unit, function
    
    # Signal for live view stop button
    live_view_stopped = pyqtSignal()


    def __init__(self):
        super().__init__()
        self.ser = serial.Serial()
        self.port = ''
        self.timeout = 0.09
        self.sep = '\t'
        self.overloads = False
        self.map_cache = {}
        self._is_running = False # For live view loop

    def start_serial(self, port, timeout):
        """
        Initializes and opens the serial port.
        """
        self.port = port
        self.timeout = timeout
        self.map_cache = {} # Clear cache for new connection
        
        if self.ser.is_open:
            self.ser.close()
            
        try:
            self.ser = serial.Serial(port=self.port,
                                    baudrate=115200, bytesize=8, parity='N', stopbits=1,
                                    timeout=self.timeout, rtscts=False, dsrdtr=False)
            return True
        except serial.serialutil.SerialException as err:
            self.error.emit(f'Serial port {self.port} does not respond: {err}')
            return False

    def data_is_ok(self, data):
        """Check if the received data block from DMM is valid."""
        # No status code yet
        if len(data) < 2: return False

        # Non-OK status
        if len(data) == 2 and chr(data[0]) == '0' and chr(data[1]) == "\r": return True

        # Non-OK status with extra data on end
        if len(data) > 2 and chr(data[0]) != '0': return False

        # We should now be in OK state
        if not data.startswith(b"0\r"): return False

        return len(data) >= 4 and chr(data[-1]) == '\r'

    def read_retry(self, cmd):
        """Read from serial port with retries."""
        retry_cmd_count = 0
        retry_read_count = 0
        data = b''
        
        if not self.ser.is_open:
            raise serial.serialutil.SerialException("Attempting to use a port that is not open")

        while retry_cmd_count < 20 and not self.data_is_ok(data):
            # *** FIX for live view stop ***
            # Check if we've been told to stop *before* writing
            if not self._is_running and cmd == "qddb":
                return data, False # Abort loop
            
            self.ser.write(cmd.encode() + b'\r')
            retry_read_count = 0 # Reset read count for each command write
            while retry_read_count < 20 and not self.data_is_ok(data):
                # *** FIX for live view stop ***
                # Abort this loop if stop is signaled
                if not self._is_running and cmd == "qddb":
                    return data, False # Abort loop
                
                bytes_read = self.ser.read(self.ser.in_waiting or 1) # Read waiting bytes or 1
                data += bytes_read
                if self.data_is_ok(data):
                    return data, True
                time.sleep(0.01) # Give device time to respond
                retry_read_count += 1
            
            retry_cmd_count += 1
            
            # Reset buffers and port if read fails
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                self.ser.close()
                self.ser.open()
            except serial.serialutil.SerialException as e:
                self.error.emit(f"Failed to reset port: {e}")
                return data, False # Give up if port fails
            time.sleep(0.01)

        return data, False

    def meter_command(self, cmd):
        """
        Send a command to the DMM and get the response.
        Handles binary and text replies.
        """
        retry_count = 0
        status = 0
        data = b''
        
        # For non-live commands, always set _is_running to True
        is_live_cmd = (cmd == "qddb")
        if not is_live_cmd:
            self._is_running = True # Ensure non-live commands can run
        
        while retry_count < 20:
            data, result_ok = self.read_retry(cmd)
            
            # Check if loop was aborted
            if not self._is_running and is_live_cmd:
                raise serial.serialutil.SerialException('Live view was stopped by user.')
            
            if data == b'':
                raise serial.serialutil.SerialException('Did not receive data from DMM')
                
            status = chr(data[0])
            if status == '0' and chr(data[1]) == '\r': break
            if result_ok: break
            retry_count += 1

        if status != '0':
            raise ValueError(f"Command: {cmd} failed. Status={status}")
        if chr(data[1]) != '\r':
            raise serial.serialutil.SerialException('Did not receive complete reply from DMM')

        binary = data[2:4] == b'#0'

        if binary:
            return data[4:-1]
        else:
            data = [i for i in data[2:-1].decode().split(',')]
            return data

    def qemap(self, map_name):
        """Query an enumeration map (qemap) from the DMM."""
        res = self.meter_command("qemap " + str(map_name))
        
        # *** FIX ***: Check for empty response which causes int() error
        if not res or not res[0]:
            self.error.emit(f"Warning: Received empty map data for '{map_name}'. Retrying cache.")
            # Attempt to use cache if available, otherwise raise
            if map_name in self.map_cache:
                return self.map_cache[map_name]
            else:
                raise ValueError(f'By app: Received empty map for {map_name} and no cache available.')

        entry_count = int(res.pop(0))
        if len(res) != entry_count * 2:
            raise ValueError('By app: Error parsing qemap')
        dmm_map = {}
        for i in range(0, len(res), 2):
            dmm_map[res[i]] = res[i + 1]
        return dmm_map

    def get_map_value(self, map_name, string, offset):
        """Get a value from a qemap, caching the map."""
        if map_name in self.map_cache:
            dmm_map = self.map_cache[map_name]
        else:
            dmm_map = self.qemap(map_name)
            self.map_cache[map_name] = dmm_map
        value = str(get_u16(string, offset))
        if value not in dmm_map:
            raise ValueError('By app: Can not find key %s in map %s' % (value, map_name))
        return dmm_map[value]

    def get_multimap_value(self, map_name, string, offset):
        """Get a value from a multi-map (qemap)."""
        if map_name in self.map_cache:
            dmm_map = self.map_cache[map_name]
        else:
            dmm_map = self.qemap(map_name)
            self.map_cache[map_name] = dmm_map
        value = str(get_u16(string, offset))
        if value not in dmm_map:
            raise ValueError('By app: Can not find key %s in map %s' % (value, map_name))
        ret = [dmm_map[value]]
        return ret

    def parse_readings(self, reading_bytes):
        """Parse a block of reading data."""
        readings = {}
        chunks, chunk_size = len(reading_bytes), 30
        list_readings = [reading_bytes[i:i + chunk_size] for i in range(0, chunks, chunk_size)]
        for r in list_readings:
            try:
                readings[self.get_map_value('readingid', r, 0)] = {
                    'value': get_double(r, 2),
                    'unit': self.get_map_value('unit', r, 10),
                    'unit_multiplier': get_s16(r, 12),
                    'decimals': get_s16(r, 14),
                    'display_digits': get_s16(r, 16),
                    'state': self.get_map_value('state', r, 18),
                    'attribute': self.get_map_value('attribute', r, 20),
                    'ts': get_time(r, 22)
                }
            except Exception as e:
                self.output.emit(f"--- Warning: Could not parse reading. {e} ---")
                continue # Skip this reading
        return readings

    def meter_id(self):
        """Get DMM ID information."""
        res = self.meter_command("ID")
        return {'model_number': res[0], 'software_version': res[1], 'serial_number': res[2]}

    def clock(self):
        """Get DMM clock time."""
        res = self.meter_command("qmp clock")
        return res[0]

    def qsls(self):
        """Query storage list summary (qsls)."""
        res = self.meter_command("qsls")
        return {'nb_recordings': res[0], 'nb_min_max': res[1], 'nb_peak': res[2], 'nb_measurements': res[3]}

    def qrsi(self, idx):
        """Query Recording Summary Information (qrsi)."""
        res = self.meter_command('qrsi ' + idx)
        reading_count = get_u16(res, 76)
        if len(res) < reading_count * 30 + 78:
            raise ValueError(
                'By app: qrsi parse error, expected at least %d bytes, got %d' % (reading_count * 30 + 78, len(res)))
        return {
            'seq_no': get_u16(res, 0),
            'un2': get_u16(res, 2),
            'start_ts': parse_time(get_double(res, 4)),
            'end_ts': parse_time(get_double(res, 12)),
            'sample_interval': get_double(res, 20),
            'event_threshold': get_double(res, 28),
            'reading_index': get_u16(res, 36),  # 32 bits?
            'un3': get_u16(res, 38),
            'num_samples': get_u16(res, 40),  # Is this 32 bits? What's in 42
            'un4': get_u16(res, 42),
            'prim_function': self.get_map_value('primfunction', res, 44),
            'sec_function': self.get_map_value('secfunction', res, 46),  # sec?
            'auto_range': self.get_map_value('autorange', res, 48),
            'unit': self.get_map_value('unit', res, 50),
            'range_max ': get_double(res, 52),
            'unit_multiplier': get_s16(res, 60),
            'bolt': self.get_map_value('bolt', res, 62),  # bolt?
            'un8': get_u16(res, 64),  # ts3?
            'un9': get_u16(res, 66),  # ts3?
            'un10': get_u16(res, 68),  # ts3?
            'un11': get_u16(res, 70),  # ts3?
            'mode': self.get_multimap_value('mode', res, 72),
            'un12': get_u16(res, 74),
            # 76 is reading count
            'readings': self.parse_readings(res[78:78 + reading_count * 30]),
            'name': res[(78 + reading_count * 30):]
        }
        
    def qsmr(self, idx):
        """Query Saved Measurement (qsmr)."""
        res = self.meter_command('qsmr ' + idx)
        reading_count = get_u16(res, 36)

        if len(res) < reading_count * 30 + 38:
            raise ValueError(
                'By app: qsmr parse error, expected at least %d bytes, got %d' % (reading_count * 30 + 78, len(res)))

        return {'[seq_no': get_u16(res, 0),
                'un1': get_u16(res, 2),  # 32 bit?
                'prim_function': self.get_map_value('primfunction', res, 4),  # prim?
                'sec_function': self.get_map_value('secfunction', res, 6),  # sec?
                'auto_range': self.get_map_value('autorange', res, 8),
                'unit': self.get_map_value('unit', res, 10),
                'range_max': get_double(res, 12),
                'unit_multiplier': get_s16(res, 20),
                'bolt': self.get_map_value('bolt', res, 22),
                'un4': get_u16(res, 24),  # ts?
                'un5': get_u16(res, 26),
                'un6': get_u16(res, 28),
                'un7': get_u16(res, 30),
                'mode': self.get_multimap_value('mode', res, 32),
                'un9': get_u16(res, 34),
                # 36 is reading count
                'readings': self.parse_readings(res[38:38 + reading_count * 30]),
                'name': res[(38 + reading_count * 30):]
                }
                
    def qsrr(self, reading_idx, sample_idx):
        """Query Saved Recording Reading (qsrr)."""
        retry_count = 0
        res = b''
        while retry_count < 20:
            res = self.meter_command("qsrr " + reading_idx + "," + sample_idx)
            if len(res) == 146:
                return {
                    'start_ts': parse_time(get_double(res, 0)),
                    'end_ts': parse_time(get_double(res, 8)),
                    'readings': self.parse_readings(res[16:16 + 30 * 3]),
                    'duration': round(get_u16(res, 106), 5),
                    'un2': get_u16(res, 108),
                    'readings2': self.parse_readings(res[110:110 + 30]),
                    'record_type': self.get_map_value('recordtype', res, 140),
                    'stable': self.get_map_value('isstableflag', res, 142),
                    'transient_state': self.get_map_value('transientstate', res, 144)
                }
            else:
                retry_count += 1
                self.output.emit(f"--- Warning: Retrying qsrr. Invalid block size: {len(res)} should be 146 ---")
                time.sleep(0.01) # Give it a moment

        raise ValueError('By app: Invalid block size: %d should be 146' % (len(res)))

    def do_min_max_cmd(self, cmd, idx):
        """Helper for MinMax and Peak commands (qmmsi, qpsi)."""
        res = self.meter_command(cmd + " " + idx)
        reading_count = get_u16(res, 52)
        if len(res) < reading_count * 30 + 54:
            raise ValueError(
                'By app: qmm/psi parse error, expected at least %d bytes, got %d' % (reading_count * 30 + 54, len(res)))

        return {'seq_no': get_u16(res, 0),
                'un2': get_u16(res, 2),
                'start_ts': parse_time(get_double(res, 4)),
                'end_ts': parse_time(get_double(res, 12)),
                'prim_function': self.get_map_value('primfunction', res, 20),
                'sec_function': self.get_map_value('secfunction', res, 22),
                'autorange': self.get_map_value('autorange', res, 24),
                'unit': self.get_map_value('unit', res, 26),
                'range_max ': get_double(res, 28),
                'unit_multiplier': get_s16(res, 36),
                'bolt': self.get_map_value('bolt', res, 38),
                'ts3': parse_time(get_double(res, 40)),
                'mode': self.get_multimap_value('mode', res, 48),
                'un8': get_u16(res, 50),
                # 52 is reading_count
                'readings': self.parse_readings(res[54:54 + reading_count * 30]),
                'name': res[(54 + reading_count * 30):]
                }
                
    def qddb(self):
        """Query Display Data Block (qddb)."""
        current_bytes = self.meter_command("qddb")

        reading_count = get_u16(current_bytes, 32)
        if len(current_bytes) != reading_count * 30 + 34:
            raise ValueError(
                'By app: qddb parse error, expected %d bytes, got %d' % ((reading_count * 30 + 34), len(current_bytes)))
        
        return {
            'prim_function': self.get_map_value('primfunction', current_bytes, 0),
            'sec_function': self.get_map_value('secfunction', current_bytes, 2),
            'auto_range': self.get_map_value('autorange', current_bytes, 4),
            'unit': self.get_map_value('unit', current_bytes, 6),
            'range_max': get_double(current_bytes, 8),
            'unit_multiplier': get_s16(current_bytes, 16),
            'bolt': self.get_map_value('bolt', current_bytes, 18),
            'ts': 0, # Not used here
            'mode': self.get_multimap_value('mode', current_bytes, 28),
            'un1': get_u16(current_bytes, 30),
            # 32 is reading count
            'readings': self.parse_readings(current_bytes[34:])
        }

    # --- Worker Slots (Callable from GUI) ---

    @pyqtSlot(str, float, str, bool, str, str)
    def do_list(self, port, timeout, kind_rec, save_to_file, file_path=None, sep=None):
        """
        Worker slot to list recordings.
        """
        f = None # File handle
        try:
            self._is_running = True # Mark as running
            if not self.start_serial(port, timeout):
                self.finished.emit()
                return

            self.sep = sep if sep else '\t'
            
            # --- File handling ---
            if save_to_file:
                try:
                    f = open(file_path, 'w', encoding='utf-8')
                    # Use a local output function to write to file OR emit
                    def write_output(msg):
                        f.write(msg + '\n')
                except Exception as e:
                    self.error.emit(f"Failed to open file: {e}")
                    self.finished.emit()
                    return
            else:
                # Local output function just emits to GUI
                def write_output(msg):
                    self.output.emit(msg)
            
            # --- Start logic ---
            
            nb = self.qsls()
            nbr = int(nb['nb_recordings'])
            nbmm = int(nb['nb_min_max'])
            nbp = int(nb['nb_peak'])

            items = {}
            if kind_rec == 'recordings':
                items[kind_rec] = {'cmd': 'qrsi', 'nb': nbr, 'lib': 'Recording'}
            elif kind_rec == 'minmax':
                items[kind_rec] = {'cmd': 'qmmsi', 'nb': nbmm, 'lib': 'MinMax'}
            elif kind_rec == 'peak':
                items[kind_rec] = {'cmd': 'qpsi', 'nb': nbp, 'lib': 'Peak'}
            else: # 'all'
                items = {'minmax': {'cmd': 'qmmsi', 'nb': nbmm, 'lib': 'MinMax'},
                         'peak': {'cmd': 'qpsi', 'nb': nbp, 'lib': 'Peak'},
                         'recordings': {'cmd': 'qrsi', 'nb': nbr, 'lib': 'Recording'}}

            for item in items:
                cmd = items[item]['cmd']
                nb = items[item]['nb']
                lib = items[item]['lib']
                
                if nb == 0:
                    write_output(f"--- No {lib} items found ---")
                    write_output("")
                    continue

                if item in ['minmax', 'peak']:
                    write_output(f"--- {lib} Items ({nb} found) ---")
                    header = ['Index', 'Name', 'Type', 'Start', 'End', 'Duration']
                    write_output(self.sep.join(header))
                    for i in range(1, nb + 1):
                        mm = self.do_min_max_cmd(cmd, str(i - 1))
                        duration = format_duration(mm['start_ts'], mm['end_ts'])
                        name = mm['name'].decode(errors='ignore')
                        debut_d = time.strftime('%Y-%m-%d %H:%M:%S', mm['start_ts'])
                        fin_d = time.strftime('%Y-%m-%d %H:%M:%S', mm['end_ts'])
                        line = [str(i), name, lib, debut_d, fin_d, duration]
                        write_output(self.sep.join(line))
                    write_output("")

                if item == 'recordings':
                    write_output(f"--- {lib} Items ({nbr} found) ---")
                    header = ['Index', 'Name', 'Type', 'Start', 'End', 'Duration', 'Measurements']
                    write_output(self.sep.join(header))
                    for i in range(1, nb + 1):
                        recording = self.qrsi(str(i - 1))
                        duration = format_duration(recording['start_ts'], recording['end_ts'])
                        name = recording['name'].decode(errors='ignore')
                        num_samples = recording['num_samples']
                        debut_d = time.strftime('%Y-%m-%d %H:%M:%S', recording['start_ts'])
                        fin_d = time.strftime('%Y-%m-%d %H:%M:%S', recording['end_ts'])
                        line = [str(i), name, lib, debut_d, fin_d, duration, str(num_samples)]
                        write_output(self.sep.join(line))
                    write_output("")

            if kind_rec == 'all':
                write_output("--- Saved Measurements ---")
                self.do_saved_measurements(records=None, write_output_func=write_output)
                write_output("")
            
            if save_to_file:
                self.output.emit(f"Successfully saved list data to {file_path}")

        except Exception as e:
            self.error.emit(f"An error occurred: {e}")
        finally:
            if self.ser.is_open:
                self.ser.close()
            if f:
                f.close()
            self._is_running = False # Mark as not running
            self.finished.emit()
            
    def do_saved_measurements(self, records, write_output_func):
        """Internal helper to list saved measurements."""
        nb_measurements = int(self.qsls()['nb_measurements'])
        interval = []
        for i in range(1, nb_measurements + 1):
            interval.append(str(i))
        
        series = records if records is not None else interval
        if not series and nb_measurements == 0:
             write_output_func(f"--- No Saved Measurements found ---")
             return

        header = ['Index', 'Name', 'Type', 'Datetime', 'Measurement', 'Unit']
        write_output_func(self.sep.join(header))
        
        found = False
        for i in series:
            if i.isdigit():
                try:
                    measurement = self.qsmr(str(int(i) - 1))
                    line = [
                        i,
                        measurement['name'].decode(errors='ignore'),
                        'Measurement',
                        time.strftime('%Y-%m-%d %H:%M:%S', measurement['readings']['PRIMARY']['ts']),
                        str(measurement['readings']['PRIMARY']['value']),
                        measurement['readings']['PRIMARY']['unit']
                    ]
                    write_output_func(self.sep.join(line))
                    found = True
                except Exception as e:
                    write_output_func(f"--- Error reading index {i}: {e} ---")
            else: # Search by name
                for j in interval:
                    try:
                        measurement = self.qsmr(str(int(j) - 1))
                        if measurement['name'] == i.encode():
                            found = True
                            line = [
                                j,
                                measurement['name'].decode(errors='ignore'),
                                'Measurement',
                                time.strftime('%Y-%m-%d %H:%M:%S', measurement['readings']['PRIMARY']['ts']),
                                str(measurement['readings']['PRIMARY']['value']),
                                measurement['readings']['PRIMARY']['unit']
                            ]
                            write_output_func(self.sep.join(line))
                            break # Found name, stop inner loop
                    except Exception:
                        continue # Error on this index, keep searching
        if not found and records is not None:
             write_output_func(f"--- Saved name(s) not found ---")

    @pyqtSlot(str, float, str, str, bool, bool, str, str)
    def do_get_data(self, port, timeout, kind_rec, indices, overloads, save_to_file, file_path, sep):
        """
        Worker slot to GET data (recordings, minmax, peak, measurements).
        """
        f = None # File handle
        try:
            self._is_running = True # Mark as running
            if not self.start_serial(port, timeout):
                self.finished.emit()
                return

            self.sep = sep if sep else '\t'
            self.overloads = overloads
            records = indices.split(",") if indices else []
            
            # --- File handling ---
            if save_to_file:
                try:
                    f = open(file_path, 'w', encoding='utf-8')
                    # Use a local output function to write to file OR emit
                    def write_output(msg):
                        f.write(msg + '\n')
                except Exception as e:
                    self.error.emit(f"Failed to open file: {e}")
                    self.finished.emit()
                    return
            else:
                # Local output function just emits to GUI
                def write_output(msg):
                    self.output.emit(msg)

            # --- Start logic ---
            if kind_rec == 'recordings':
                self.get_recordings_data(records, write_output)
            elif kind_rec == 'minmax':
                self.get_min_max_peak_data(records, 'nb_min_max', 'qmmsi', write_output)
            elif kind_rec == 'peak':
                self.get_min_max_peak_data(records, 'nb_peak', 'qpsi', write_output)
            elif kind_rec == 'measurements':
                 write_output("--- Getting Saved Measurements ---")
                 self.do_saved_measurements(records, write_output_func=write_output)

            if save_to_file:
                self.output.emit(f"Successfully saved data to {file_path}")

        except Exception as e:
            self.error.emit(f"An error occurred: {e}")
        finally:
            if self.ser.is_open:
                self.ser.close()
            if f:
                f.close()
            self._is_running = False # Mark as not running
            self.finished.emit()

    def get_recordings_data(self, records, write_output):
        """Internal helper to get full recordings data."""
        nb_recordings = int(self.qsls()['nb_recordings'])
        interval = [str(i) for i in range(1, nb_recordings + 1)]
        series = records if records else interval
        
        if not series:
            write_output("--- No recordings to fetch ---")
            return

        found = False
        for i in series:
            record_index_str = ""
            if i.isdigit():
                record_index_str = str(int(i) - 1)
                found = True # Assume it will be found
            else:
                # Search for the name
                for j in interval:
                    try:
                        recording = self.qrsi(str(int(j) - 1))
                        if recording['name'] == i.encode():
                            record_index_str = str(int(j) - 1)
                            found = True
                            break
                    except Exception:
                        continue # Error on this index, keep searching
            
            if not record_index_str:
                if i.strip(): # Only show error if 'i' wasn't just whitespace
                    write_output(f"--- Recording '{i}' not found ---")
                continue # Not found or not a digit
            
            # --- Process the found recording ---
            try:
                recording = self.qrsi(record_index_str)
                duration = format_duration(recording['start_ts'], recording['end_ts'])
                write_output('Index %s, Name %s, Start %s, End %s, Duration %s, Measurements %s'
                      % (str(int(record_index_str)+1), (recording['name']).decode(errors='ignore'),
                         time.strftime('%Y-%m-%d %H:%M:%S', recording['start_ts']),
                         time.strftime('%Y-%m-%d %H:%M:%S', recording['end_ts']), 
                         duration, recording['num_samples']))
                
                header = ['Start Time', 'Primary', '', 'Maximum', '', 'Average', '', 'Minimum', '', '#Samples', 'Type']
                write_output(self.sep.join(header))

                for k in range(0, recording['num_samples']):
                    measurement = self.qsrr(str(recording['reading_index']), str(k))
                    
                    if self.overloads and \
                            (measurement['readings2']['PRIMARY']['value'] == 9.99999999e+37 or
                             measurement['readings']['MAXIMUM']['value'] == 9.99999999e+37 or
                             measurement['readings']['MINIMUM']['value'] == 9.99999999e+37):
                        continue
                    
                    avg_duration = 0.0
                    try:
                        avg_duration = measurement['readings']['AVERAGE']['value'] / measurement['duration']
                    except (ZeroDivisionError, TypeError):
                        pass # avg_duration remains 0
                        
                    duration_str = str(round(avg_duration, measurement['readings']['AVERAGE']['decimals']))

                    line = [
                        time.strftime('%Y-%m-%d %H:%M:%S', measurement['start_ts']),
                        str(measurement['readings2']['PRIMARY']['value']),
                        measurement['readings2']['PRIMARY']['unit'],
                        str(measurement['readings']['MAXIMUM']['value']),
                        measurement['readings']['MAXIMUM']['unit'],
                        duration_str,
                        measurement['readings']['AVERAGE']['unit'],
                        str(measurement['readings']['MINIMUM']['value']),
                        measurement['readings']['MINIMUM']['unit'],
                        str(measurement['duration']),
                        'INTERVAL' if measurement['record_type'] == 'INTERVAL' else measurement['stable']
                    ]
                    write_output(self.sep.join(line))
                write_output("") # Newline after each recording
            
            except Exception as e:
                write_output(f"--- Error processing index {i}: {e} ---")

        if not found and records:
            write_output("--- None of the specified saved names were found ---")
            
    def get_min_max_peak_data(self, records, field, cmd, write_output):
        """Internal helper to get full Min/Max or Peak data."""
        nb_items = int(self.qsls()[field])
        interval = [str(i) for i in range(1, nb_items + 1)]
        series = records if records else interval
        
        if not series:
            write_output(f"--- No {cmd} items to fetch ---")
            return

        found = False
        for i in series:
            item_index_str = ""
            if i.isdigit():
                item_index_str = str(int(i) - 1)
                found = True # Assume it will be found
            else:
                # Search for the name
                for j in interval:
                    try:
                        measurement = self.do_min_max_cmd(cmd, str(int(j) - 1))
                        if measurement['name'] == i.encode():
                            item_index_str = str(int(j) - 1)
                            found = True
                            break
                    except Exception:
                        continue
            
            if not item_index_str:
                if i.strip():
                    write_output(f"--- Item '{i}' not found ---")
                continue

            # --- Process the found item ---
            try:
                measurement = self.do_min_max_cmd(cmd, item_index_str)
                
                write_output(f"--- {(measurement['name']).decode(errors='ignore')} (Index {str(int(item_index_str)+1)}) ---")
                write_output(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S', measurement['start_ts'])}")
                write_output(f"End:   {time.strftime('%Y-%m-%d %H:%M:%S', measurement['end_ts'])}")
                write_output(f"Mode:  {measurement['autorange']}, Range {int(measurement['range_max '])} {measurement['unit']}")
                write_output("") # newline
                
                self.print_min_max_peak_detail(measurement, 'PRIMARY', write_output)
                self.print_min_max_peak_detail(measurement, 'MAXIMUM', write_output)
                self.print_min_max_peak_detail(measurement, 'AVERAGE', write_output)
                self.print_min_max_peak_detail(measurement, 'MINIMUM', write_output)
                write_output("") # newline

            except Exception as e:
                write_output(f"--- Error processing index {i}: {e} ---")
                
        if not found and records:
            write_output("--- None of the specified saved names were found ---")

    def print_min_max_peak_detail(self, measurement, detail, write_output):
        """Helper to format Min/Max/Peak lines."""
        if detail in measurement['readings']:
            line = [
                f"{detail:<10}",
                str(measurement['readings'][detail]['value']),
                measurement['readings'][detail]['unit'],
                time.strftime('%Y-%m-%d %H:%M:%S', measurement['readings'][detail]['ts'])
            ]
            write_output(self.sep.join(line))
        else:
             write_output(f"{detail:<10}{self.sep}--- N/A ---")

    @pyqtSlot(str, float)
    def get_config(self, port, timeout):
        """Worker slot to get DMM configuration."""
        try:
            self._is_running = True # Mark as running
            if not self.start_serial(port, timeout):
                self.finished.emit()
                return

            self.output.emit("--- Device Configuration ---")
            info = self.meter_id()
            self.output.emit(f"Model:{self.sep}{info['model_number']}")
            self.output.emit(f"Software Version:{self.sep}{info['software_version']}")
            self.output.emit(f"Serial Number:{self.sep}{info['serial_number']}")
            
            try:
                current_time = time.gmtime(int(self.clock()))
                self.output.emit(f"Current meter time:{self.sep}{time.strftime('%Y-%m-%d %H:%M:%S', current_time)}")
            except Exception as e:
                self.output.emit(f"Current meter time:{self.sep}Error: {e}")

            # Define properties to get
            props = [
                "company", "contact", "operator", "site", 
                "aheventTh", "lang", "dateFmt", "timeFmt", 
                "digits", "beeper", "tempOS", "numFmt", 
                "ablto", "apoffto"
            ]
            
            # Map for friendlier names
            prop_names = {
                "aheventTh": "Autohold Threshold",
                "lang": "Language",
                "dateFmt": "Date Format",
                "timeFmt": "Time Format",
                "digits": "Digits",
                "beeper": "Beeper",
                "tempOS": "Temperature Offset Shift",
                "numFmt": "Numeric Format",
                "ablto": "Auto Backlight Timeout",
                "apoffto": "Auto Power Off"
            }

            for prop in props:
                try:
                    val = self.meter_command(f"qmpq {prop}")[0].lstrip("'").rstrip("'")
                    name = prop_names.get(prop, prop.capitalize())
                    self.output.emit(f"{name}:{self.sep}{val}")
                except Exception as e:
                    self.output.emit(f"{prop.capitalize()}:{self.sep}Error: {e}")
            
            self.output.emit("\n--- Storable Names ---")
            # Call get_names *without* emitting finished signal
            self.get_names_internal()

        except Exception as e:
            self.error.emit(f"An error occurred: {e}")
        finally:
            if self.ser.is_open:
                self.ser.close()
            self._is_running = False # Mark as not running
            self.finished.emit()

    def get_names_internal(self):
        """Internal helper for get_names to be called by other slots."""
        # Assumes serial port is already open
        try:
            self.output.emit(f"Index{self.sep}Name")
            for i in range(1, 9):
                cmd = 'qsavname ' + str(i - 1)
                res = self.meter_command(cmd)
                self.output.emit(f"{i}{self.sep}{res[0].split(chr(13))[0]}") # Split on \r
        except Exception as e:
            self.error.emit(f"An error occurred getting names: {e}")


    @pyqtSlot(str, float)
    def get_names(self, port, timeout):
        """Worker slot to get storable names (legacy, not used by GUI)."""
        # This is kept for the old trigger, but get_config is better
        try:
            self._is_running = True # Mark as running
            if not self.start_serial(port, timeout):
                self.finished.emit()
                return
            self.get_names_internal()
        except Exception as e:
            self.error.emit(f"An error occurred: {e}")
        finally:
            if self.ser.is_open:
                self.ser.close()
            self._is_running = False # Mark as not running
            self.finished.emit()

    #@pyqtSlot(str, float, str, str) # Removed decorator to fix signature match
    def get_property(self, port, timeout, prop_name, prop_type):
        """Worker slot to get a single DMM property."""
        try:
            self._is_running = True # Mark as running
            if not self.start_serial(port, timeout):
                self.finished.emit()
                return

            if prop_type == 'qmpq':
                val = self.meter_command(f"qmpq {prop_name}")[0].lstrip("'").rstrip("'")
                self.output.emit(f"Value for '{prop_name}': {val}")
            elif prop_type == 'qsavname':
                # prop_name is the index
                idx = int(prop_name) - 1
                if 0 <= idx <= 7:
                    val = self.meter_command(f"qsavname {idx}")[0].split(chr(13))[0]
                    self.output.emit(f"Value for name index {prop_name}: {val}")
                else:
                    self.error.emit("Index must be between 1 and 8")

        except Exception as e:
            self.error.emit(f"An error occurred: {e}")
        finally:
            if self.ser.is_open:
                self.ser.close()
            self._is_running = False # Mark as not running
            self.finished.emit()

    @pyqtSlot(str, float, str, str, str)
    def set_property(self, port, timeout, prop_name, prop_value, prop_type):
        """Worker slot to set a DMM property."""
        try:
            self._is_running = True # Mark as running
            if not self.start_serial(port, timeout):
                self.finished.emit()
                return
                
            cmd = ""
            if prop_type == 'qmpq':
                # Add quotes around value for mpq command
                cmd = f"mpq {prop_name},'{prop_value}'"
            elif prop_type == 'savname':
                # prop_name is the index
                idx = int(prop_name) - 1
                if 0 <= idx <= 7:
                    # Add quotes around value for savname command
                    cmd = f'savname {idx},"{prop_value}"'
                else:
                    self.error.emit("Index must be between 1 and 8")
                    self.finished.emit()
                    return
            
            if cmd:
                self.meter_command(cmd)
                self.output.emit(f"Successfully set {prop_name} to '{prop_value}'")
            else:
                self.error.emit("Invalid property type")

        except Exception as e:
            self.error.emit(f"An error occurred: {e}")
        finally:
            if self.ser.is_open:
                self.ser.close()
            self._is_running = False # Mark as not running
            self.finished.emit()
            
    def do_sync_time(self):
        """Internal helper to sync time."""
        # This function assumes start_serial() has been called
        lt = calendar.timegm(datetime.datetime.now().utctimetuple())
        cmd = 'mp clock,' + str(lt)
        self.ser.write(cmd.encode() + b'\r')
        time.sleep(0.1)
        res = self.ser.read(2) # Read the '0\r' response
        if res == b'0\r':
            self.output.emit("Successfully synced the clock of the DMM")
        else:
            self.error.emit(f"Clock sync failed. Response: {res}")
            
    @pyqtSlot(str, float)
    def sync_time(self, port, timeout):
        """Worker slot to sync DMM time to PC time."""
        try:
            self._is_running = True # Mark as running
            if not self.start_serial(port, timeout): # Must start serial first
                self.finished.emit()
                return
            
            self.do_sync_time()

        except Exception as e:
            self.error.emit(f"An error occurred: {e}")
        finally:
            if self.ser.is_open:
                self.ser.close()
            self._is_running = False # Mark as not running
            self.finished.emit()
    
    @pyqtSlot(str, float)
    def start_live_view(self, port, timeout):
        """Worker slot to start streaming live data."""
        self._is_running = True
        try:
            if not self.start_serial(port, timeout):
                self.live_view_stopped.emit() # Signal stop if port fails
                return
            
            self.output.emit("--- Live View Started (Press 'Stop' to end) ---")
            
            while self._is_running:
                try:
                    res = self.qddb()
                    # Check for LIVE reading, fallback to PRIMARY
                    live_reading = res['readings'].get('LIVE', res['readings'].get('PRIMARY'))
                    
                    if not live_reading:
                        self.live_data.emit("---", "N/A", "Waiting...")
                        time.sleep(0.5)
                        continue

                    val = str(live_reading['value'])
                    unit = live_reading['unit']
                    func = res['prim_function']
                    self.live_data.emit(val, unit, func)
                
                except Exception as e:
                    # *** FIX for live view stop ***
                    # If we stopped, just break silently.
                    if not self._is_running:
                        break 
                        
                    # Otherwise, report the read error
                    self.output.emit(f"--- Live View Read Error: {e} ---")
                    self.live_data.emit("Error", "---", "See Log")
                    
                    # Break if port is closed
                    if not self.ser.is_open:
                        self.output.emit("--- Port closed, stopping ---")
                        break
                    time.sleep(1.0) # Pause after an error
                
                # Check for stop signal every 100ms
                for _ in range(5): # Check more often for responsiveness
                    if not self._is_running:
                        break
                    time.sleep(0.1) # Shorter sleep for a ~0.5s update rate
                
        except Exception as e:
            # If we are stopping, don't report the "stopped by user" error
            if self._is_running:
                self.error.emit(f"An error occurred: {e}")
        finally:
            if self.ser.is_open:
                self.ser.close()
            self.output.emit("--- Live View Stopped ---")
            self.live_data.emit("---", "---", "Stopped")
            self.live_view_stopped.emit() # Signal GUI to re-enable button
            self._is_running = False

    @pyqtSlot()
    def stop_live_view(self):
        """Slot to stop the live view loop."""
        self._is_running = False


class DMMUtilApp(QMainWindow):
    """
    The main GUI window.
    """
    
    # --- Signals to Worker Thread ---
    trigger_list = pyqtSignal(str, float, str, bool, str, str)
    trigger_get_data = pyqtSignal(str, float, str, str, bool, bool, str, str)
    trigger_get_config = pyqtSignal(str, float)
    trigger_get_names = pyqtSignal(str, float) # Keep for compatibility, though unused
    trigger_get_property = pyqtSignal(str, float, str, str)
    trigger_set_property = pyqtSignal(str, float, str, str, str)
    trigger_sync_time = pyqtSignal(str, float)
    trigger_start_live = pyqtSignal(str, float)
    trigger_stop_live = pyqtSignal()
    

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fluke 28x DMM Utility")
        self.setGeometry(100, 100, 800, 700) # Increased height for live readout
        
        self.init_ui()
        self.init_worker()
        
        self.refresh_ports_list()
        self.show()

    def init_worker(self):
        """Set up the worker thread for DMM communication."""
        self.worker_thread = QThread()
        self.worker = DMMWorker()
        self.worker.moveToThread(self.worker_thread)

        # Connect worker signals to GUI slots
        self.worker.output.connect(self.append_output)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.on_task_finished)
        self.worker.live_data.connect(self.update_live_readout)
        self.worker.live_view_stopped.connect(self.on_live_view_stopped)
        
        # Connect GUI triggers to worker slots
        self.trigger_list.connect(self.worker.do_list)
        self.trigger_get_data.connect(self.worker.do_get_data)
        self.trigger_get_config.connect(self.worker.get_config)
        self.trigger_get_names.connect(self.worker.get_names) # Kept for compatibility
        self.trigger_get_property.connect(self.worker.get_property)
        self.trigger_set_property.connect(self.worker.set_property)
        self.trigger_sync_time.connect(self.worker.sync_time)
        self.trigger_start_live.connect(self.worker.start_live_view)
        self.trigger_stop_live.connect(self.worker.stop_live_view)
        
        # Start the thread
        self.worker_thread.start()
        
    def init_ui(self):
        """Set up the main user interface."""
        
        # --- Main Widget and Layout ---
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # --- Top: Connection Settings ---
        conn_group = QGroupBox("Connection")
        conn_layout = QGridLayout()
        conn_group.setLayout(conn_layout)
        
        conn_layout.addWidget(QLabel("Serial Port:"), 0, 0)
        self.port_combo = QComboBox()
        conn_layout.addWidget(self.port_combo, 0, 1)
        
        self.refresh_ports_btn = QPushButton("Refresh")
        self.refresh_ports_btn.clicked.connect(self.refresh_ports_list)
        conn_layout.addWidget(self.refresh_ports_btn, 0, 2)
        
        conn_layout.addWidget(QLabel("Timeout (s):"), 1, 0)
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setValue(0.09)
        self.timeout_spin.setSingleStep(0.01)
        self.timeout_spin.setDecimals(2)
        conn_layout.addWidget(self.timeout_spin, 1, 1)
        
        main_layout.addWidget(conn_group)
        
        # --- Middle: Tabs for Actions ---
        self.tabs = QTabWidget()
        
        # Tab 1: List & Get Data
        self.data_tab = QWidget()
        data_layout = QVBoxLayout()
        self.data_tab.setLayout(data_layout)
        
        list_group = QGroupBox("List Data Summaries")
        list_layout = QHBoxLayout()
        list_group.setLayout(list_layout)
        
        self.list_recordings_btn = QPushButton("List Recordings")
        self.list_recordings_btn.clicked.connect(lambda: self.run_list_task('recordings'))
        self.list_minmax_btn = QPushButton("List Min/Max")
        self.list_minmax_btn.clicked.connect(lambda: self.run_list_task('minmax'))
        self.list_peak_btn = QPushButton("List Peak")
        self.list_peak_btn.clicked.connect(lambda: self.run_list_task('peak'))
        self.list_all_btn = QPushButton("List All")
        self.list_all_btn.clicked.connect(lambda: self.run_list_task('all'))
        
        list_layout.addWidget(self.list_recordings_btn)
        list_layout.addWidget(self.list_minmax_btn)
        list_layout.addWidget(self.list_peak_btn)
        list_layout.addWidget(self.list_all_btn)
        data_layout.addWidget(list_group)
        
        get_group = QGroupBox("Get Full Data")
        get_layout = QGridLayout()
        get_group.setLayout(get_layout)
        
        get_layout.addWidget(QLabel("Data Type:"), 0, 0)
        self.get_type_combo = QComboBox()
        self.get_type_combo.addItems(["Recordings", "Min/Max", "Peak", "Measurements"])
        get_layout.addWidget(self.get_type_combo, 0, 1)
        
        get_layout.addWidget(QLabel("Indices or Names:"), 1, 0)
        self.indices_edit = QLineEdit()
        self.indices_edit.setPlaceholderText("e.g., 1,3,MyRec (blank for all)")
        get_layout.addWidget(self.indices_edit, 1, 1)
        
        self.overloads_check = QCheckBox("Hide Overloads (recordings only)")
        get_layout.addWidget(self.overloads_check, 2, 1)
        
        self.get_data_btn = QPushButton("Get Data to Screen")
        self.get_data_btn.clicked.connect(self.run_get_task)
        get_layout.addWidget(self.get_data_btn, 3, 0)
        
        self.save_data_btn = QPushButton("Save Data to File...")
        self.save_data_btn.clicked.connect(self.run_save_task)
        get_layout.addWidget(self.save_data_btn, 3, 1)
        
        data_layout.addWidget(get_group)
        self.tabs.addTab(self.data_tab, "Data")
        
        # Tab 2: Live View
        self.live_tab = QWidget()
        live_layout = QVBoxLayout()
        self.live_tab.setLayout(live_layout)
        
        live_group = QGroupBox("Live Measurement")
        live_group_layout = QHBoxLayout()
        live_group.setLayout(live_group_layout)
        
        self.start_live_btn = QPushButton("Start Live View")
        self.start_live_btn.clicked.connect(self.on_start_live_view)
        self.stop_live_btn = QPushButton("Stop Live View")
        self.stop_live_btn.clicked.connect(self.on_stop_live_view)
        self.stop_live_btn.setEnabled(False)
        
        live_group_layout.addWidget(self.start_live_btn)
        live_group_layout.addWidget(self.stop_live_btn)
        live_layout.addWidget(live_group)
        
        # Live Readout Display
        readout_group = QGroupBox("Current Reading")
        readout_layout = QVBoxLayout()
        readout_group.setLayout(readout_layout)

        self.live_readout_value = QLabel("---")
        self.live_readout_value.setFont(QFont("Monospace", 48, QFont.Bold)) # Monospace fallback
        self.live_readout_value.setAlignment(Qt.AlignCenter)
        self.live_readout_value.setStyleSheet("QLabel { background-color: #333; color: #0F0; padding: 10px; border-radius: 5px; }")

        self.live_readout_unit = QLabel("---")
        self.live_readout_unit.setFont(QFont("Monospace", 24))
        self.live_readout_unit.setAlignment(Qt.AlignCenter)
        self.live_readout_unit.setStyleSheet("QLabel { color: #BBB; }")

        self.live_readout_func = QLabel("Stopped")
        self.live_readout_func.setFont(QFont("Arial", 12))
        self.live_readout_func.setAlignment(Qt.AlignCenter)
        self.live_readout_func.setStyleSheet("QLabel { color: #AAA; }")

        readout_layout.addWidget(self.live_readout_value)
        readout_layout.addWidget(self.live_readout_unit)
        readout_layout.addWidget(self.live_readout_func)
        
        live_layout.addWidget(readout_group)
        live_layout.addStretch()
        self.tabs.addTab(self.live_tab, "Live View")

        # Tab 3: Device Properties
        self.config_tab = QWidget()
        config_layout = QVBoxLayout()
        self.config_tab.setLayout(config_layout)

        # General Config
        config_group = QGroupBox("General")
        config_group_layout = QHBoxLayout()
        config_group.setLayout(config_group_layout)
        self.get_config_btn = QPushButton("Get Full Configuration")
        self.get_config_btn.clicked.connect(self.on_get_config)
        self.sync_time_btn = QPushButton("Sync PC Time to DMM")
        self.sync_time_btn.clicked.connect(self.on_sync_time)
        config_group_layout.addWidget(self.get_config_btn)
        config_group_layout.addWidget(self.sync_time_btn)
        config_layout.addWidget(config_group)

        # Specific Properties
        props_group = QGroupBox("Device Properties")
        props_layout = QGridLayout()
        props_group.setLayout(props_layout)

        self.prop_combo = QComboBox()
        self.prop_combo.addItems(["Company", "Operator", "Site", "Contact"])
        self.prop_value_edit = QLineEdit()
        self.get_prop_btn = QPushButton("Get")
        self.set_prop_btn = QPushButton("Set")
        
        self.get_prop_btn.clicked.connect(self.on_get_property)
        self.set_prop_btn.clicked.connect(self.on_set_property)
        
        props_layout.addWidget(QLabel("Property:"), 0, 0)
        props_layout.addWidget(self.prop_combo, 0, 1)
        props_layout.addWidget(self.get_prop_btn, 0, 2)
        props_layout.addWidget(QLabel("Value:"), 1, 0)
        props_layout.addWidget(self.prop_value_edit, 1, 1)
        props_layout.addWidget(self.set_prop_btn, 1, 2)
        config_layout.addWidget(props_group)
        
        # Storable Names
        names_group = QGroupBox("Storable Names (1-8)")
        names_layout = QGridLayout()
        names_group.setLayout(names_layout)
        
        self.name_index_spin = QSpinBox()
        self.name_index_spin.setRange(1, 8)
        self.name_value_edit = QLineEdit()
        self.get_name_btn = QPushButton("Get Name")
        self.set_name_btn = QPushButton("Set Name")
        
        self.get_name_btn.clicked.connect(self.on_get_name)
        self.set_name_btn.clicked.connect(self.on_set_name)
        
        names_layout.addWidget(QLabel("Index:"), 0, 0)
        names_layout.addWidget(self.name_index_spin, 0, 1)
        names_layout.addWidget(self.get_name_btn, 0, 2)
        names_layout.addWidget(QLabel("Name:"), 1, 0)
        names_layout.addWidget(self.name_value_edit, 1, 1)
        names_layout.addWidget(self.set_name_btn, 1, 2)
        
        config_layout.addWidget(names_group)
        config_layout.addStretch()
        self.tabs.addTab(self.config_tab, "Device Properties")

        main_layout.addWidget(self.tabs)
        
        # --- Bottom: Output ---
        output_group = QGroupBox("Output Log")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Monospace", 9)) # Monospace font
        self.output_text.setLineWrapMode(QTextEdit.NoWrap) # Disable line wrap
        self.output_text.setMaximumHeight(200) # Set a max height for the log
        
        self.clear_output_btn = QPushButton("Clear Output Log")
        self.clear_output_btn.clicked.connect(self.output_text.clear)
        
        output_layout.addWidget(self.output_text)
        output_layout.addWidget(self.clear_output_btn)
        
        main_layout.addWidget(output_group)
        
        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Select port and action.")

    # --- GUI Slots ---

    def set_controls_enabled(self, enabled):
        """Enable or disable all UI controls *except* live view buttons."""
        # Enable/disable connection widgets
        self.port_combo.setEnabled(enabled)
        self.timeout_spin.setEnabled(enabled)
        self.refresh_ports_btn.setEnabled(enabled)
        
        # Enable/disable all tabs
        self.tabs.setTabEnabled(self.tabs.indexOf(self.data_tab), enabled)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.config_tab), enabled)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.live_tab), enabled)
        
        # *** FIX ***: This function ONLY manages non-live buttons.
        # Reset live view buttons to default state (Start=on, Stop=off)
        self.start_live_btn.setEnabled(enabled)
        self.stop_live_btn.setEnabled(False)

    def refresh_ports_list(self):
        """Find and list available serial ports."""
        self.port_combo.clear()
        ports = comports()
        if not ports:
            self.port_combo.addItem("No ports found")
            self.port_combo.setEnabled(False)
        else:
            port_list = [p.device for p in ports]
            self.port_combo.addItems(port_list)
            self.port_combo.setEnabled(True)

    @pyqtSlot(str)
    def append_output(self, text):
        """Append text to the output box."""
        self.output_text.append(text)
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )
        
    @pyqtSlot(str, str, str)
    def update_live_readout(self, value, unit, function):
        """Update the dedicated live view labels."""
        self.live_readout_value.setText(value)
        self.live_readout_unit.setText(unit)
        self.live_readout_func.setText(function)

    @pyqtSlot(str)
    def show_error(self, error_text):
        """Show an error message in a popup and in the status bar."""
        self.append_output(f"--- ERROR ---\n{error_text}\n-----------")
        self.status_bar.showMessage(f"Error: {error_text}", 5000)
        QMessageBox.critical(self, "Error", error_text)

    @pyqtSlot()
    def on_task_finished(self):
        """Called when a worker task (non-live) is complete."""
        self.status_bar.showMessage("Task finished.", 3000)
        self.set_controls_enabled(True)
        
    def get_common_params(self):
        """Get the selected port and timeout."""
        port = self.port_combo.currentText()
        if not port or "No ports" in port:
            self.show_error("No serial port selected.")
            return None, None
        timeout = self.timeout_spin.value()
        return port, timeout

    def run_list_task(self, kind):
        """Trigger the worker to list data."""
        port, timeout = self.get_common_params()
        if not port: return
        
        self.set_controls_enabled(False)
        self.status_bar.showMessage(f"Listing {kind} data...")
        self.output_text.clear()
        
        # Trigger the list task
        self.trigger_list.emit(port, timeout, kind, False, "", "") # False for save_to_file

    def run_get_task(self):
        """Trigger the worker to get data to the screen."""
        port, timeout = self.get_common_params()
        if not port: return
        
        kind_map = {
            "Recordings": "recordings",
            "Min/Max": "minmax",
            "Peak": "peak",
            "Measurements": "measurements"
        }
        
        kind = kind_map[self.get_type_combo.currentText()]
        indices = self.indices_edit.text()
        overloads = self.overloads_check.isChecked()
        sep = '\t' # Use tab for screen output
        
        self.set_controls_enabled(False)
        self.status_bar.showMessage(f"Getting {kind} data...")
        self.output_text.clear()
        
        self.trigger_get_data.emit(port, timeout, kind, indices, overloads, False, "", sep)

    def run_save_task(self):
        """Trigger the worker to save data to a file."""
        port, timeout = self.get_common_params()
        if not port: return
        
        # --- Ask for file path ---
        file_filter = "CSV (Comma-separated) (*.csv);;TSV (Tab-separated) (*.tsv);;Text File (*.txt)"
        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save Data As", "", file_filter)
        
        if not file_path:
            return # User canceled
            
        # Determine separator from filter
        if "csv" in selected_filter:
            sep = ","
        else: # tsv or txt
            sep = "\t"
        
        # --- Get other params ---
        kind_map = {
            "Recordings": "recordings",
            "Min/Max": "minmax",
            "Peak": "peak",
            "Measurements": "measurements"
        }
        kind = kind_map[self.get_type_combo.currentText()]
        indices = self.indices_edit.text()
        overloads = self.overloads_check.isChecked()

        self.set_controls_enabled(False)
        self.status_bar.showMessage(f"Saving {kind} data to {file_path}...")
        self.output_text.clear()
        self.append_output(f"Saving {kind} data to file: {file_path}\nThis may take a moment...")
        
        self.trigger_get_data.emit(port, timeout, kind, indices, overloads, True, file_path, sep)

    def on_get_config(self):
        """Trigger the worker to get DMM configuration."""
        port, timeout = self.get_common_params()
        if not port: return
        
        self.set_controls_enabled(False)
        self.status_bar.showMessage("Getting configuration...")
        self.output_text.clear()
        self.trigger_get_config.emit(port, timeout)
        
    def on_get_property(self):
        """Trigger worker to get a single property."""
        port, timeout = self.get_common_params()
        if not port: return
        
        prop_name = self.prop_combo.currentText().lower()
        
        self.set_controls_enabled(False)
        self.status_bar.showMessage(f"Getting property {prop_name}...")
        self.output_text.clear()
        self.trigger_get_property.emit(port, timeout, prop_name, 'qmpq')

    def on_set_property(self):
        """Trigger worker to set a single property."""
        port, timeout = self.get_common_params()
        if not port: return
        
        prop_name = self.prop_combo.currentText().lower()
        prop_value = self.prop_value_edit.text()
        
        if not prop_value:
            self.show_error("Value cannot be empty.")
            return
        
        self.set_controls_enabled(False)
        self.status_bar.showMessage(f"Setting property {prop_name}...")
        self.output_text.clear()
        self.trigger_set_property.emit(port, timeout, prop_name, prop_value, 'qmpq')

    def on_get_name(self):
        """Trigger worker to get a storable name."""
        port, timeout = self.get_common_params()
        if not port: return
        
        prop_name = str(self.name_index_spin.value()) # Index
        
        self.set_controls_enabled(False)
        self.status_bar.showMessage(f"Getting name index {prop_name}...")
        self.output_text.clear()
        self.trigger_get_property.emit(port, timeout, prop_name, 'qsavname')
        
    def on_set_name(self):
        """Trigger worker to set a storable name."""
        port, timeout = self.get_common_params()
        if not port: return
        
        prop_name = str(self.name_index_spin.value()) # Index
        prop_value = self.name_value_edit.text()
        
        if not prop_value:
            self.show_error("Name value cannot be empty.")
            return
            
        self.set_controls_enabled(False)
        self.status_bar.showMessage(f"Setting name index {prop_name}...")
        self.output_text.clear()
        self.trigger_set_property.emit(port, timeout, prop_name, prop_value, 'savname')
        
    def on_sync_time(self):
        """Trigger worker to sync DMM time."""
        port, timeout = self.get_common_params()
        if not port: return
        
        reply = QMessageBox.question(self, "Confirm Time Sync",
                                     "This will set the DMM's clock to your PC's current time. Continue?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.set_controls_enabled(False)
            self.status_bar.showMessage("Syncing time...")
            self.output_text.clear()
            self.trigger_sync_time.emit(port, timeout)

    def on_start_live_view(self):
        """Trigger worker to start live view."""
        port, timeout = self.get_common_params()
        if not port: return
        
        # *** FIX ***: Switch to the live tab *before* disabling other tabs
        self.tabs.setCurrentWidget(self.live_tab)
        
        # Disable all controls
        self.set_controls_enabled(False)
        
        # Explicitly manage live view buttons
        self.start_live_btn.setEnabled(False)
        self.stop_live_btn.setEnabled(True)
        
        # Enable *only* the live view tab
        self.tabs.setTabEnabled(self.tabs.indexOf(self.live_tab), True)
        
        self.status_bar.showMessage("Starting live view... (Click Stop to end)")
        self.output_text.clear()
        self.trigger_start_live.emit(port, timeout)

    @pyqtSlot()
    def on_stop_live_view(self):
        """Trigger worker to stop live view."""
        # Only trigger stop if the button is enabled (i.e., view is running)
        if self.stop_live_btn.isEnabled():
            self.status_bar.showMessage("Stopping live view...")
            self.stop_live_btn.setEnabled(False) # Disable stop, but don't re-enable start yet
            self.trigger_stop_live.emit()

    @pyqtSlot()
    def on_live_view_stopped(self):
        """Called by worker when live view is fully stopped."""
        # This is the *only* place that should re-enable controls after live view
        self.set_controls_enabled(True) 
        self.status_bar.showMessage("Live view stopped.", 3000)
        # set_controls_enabled(True) already handles resetting the buttons
        # to Start=Enabled, Stop=Disabled

    def closeEvent(self, event):
        """Handle window close event."""
        self.on_stop_live_view() # Stop live view if running
        self.worker_thread.quit()
        self.worker_thread.wait() # Wait for thread to finish
        event.accept()

# --- Main execution ---
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    
    # Set a more modern style if available
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    
    main_win = DMMUtilApp()
    sys.exit(app.exec_())