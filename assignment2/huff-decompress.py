import argparse
import os
import pickle
from collections import deque
from contextlib import contextmanager
import time

#psuedo eof symbol
#an integer cannot actually occur as a token (they are always a string), so it
#should never be confused for a real token
EOF_SYMBOL = -1

class InputBitstream:
    """Class to wrap an io object abstracting bitwise operations in the reading of bits"""
    def __init__(self,file):
        self.file = file
        self.buffer = deque()
        self.bit_buffer = deque()
        self.buffer.extend(self.file.read())
        
    def buffer_up(self,n):
        while len(self.bit_buffer) < n:
            byte = self.buffer.popleft()
            bits = []
            for i in range(0,8):
                bits.append(byte & 1)
                byte = byte >> 1
            bits.reverse()
            self.bit_buffer.extend(bits)
        
    def popleft(self):
        if len(self.bit_buffer) == 0:
            self.buffer_up(1)
        return self.bit_buffer.popleft()

@contextmanager
def input_bitstream(file_name):
    with open(file_name,'rb') as file:
        input_bitstream = InputBitstream(file)
        yield input_bitstream
        
@contextmanager            
def bench_section(section_name,active):
    if active:
        start = time.perf_counter()
    yield
    if active:
        end = time.perf_counter()
        print(f'Section {section_name} took {end-start} seconds')
        
class HNodeLeaf:
    def __init__(self,value=None):
        self.value = value
        
    def decode(self,bitstream):
        return self.value
                
    def make_lut(self,bits=[],lut={}):
        """Turns a huffman tree into a dict based lut which should be efficient for compression"""
        lut[self.value] = bits
        return lut

class HNode:
    def __init__(self,left=None,right=None):
        self.left = left
        self.right = right
        
    def decode(self,bitstream):
        bit = bitstream.popleft()
        if bit == 0:
            return self.left.decode(bitstream)
        else:
            return self.right.decode(bitstream)
                
    def make_lut(self,bits=[],lut={}):
        """Turns a huffman tree into a dict based lut which should be efficient for compression"""
        l_bits = bits.copy()
        l_bits.append(0)
        r_bits = bits.copy()
        r_bits.append(1)
        self.left.make_lut(bits=l_bits,lut=lut)
        self.right.make_lut(bits=r_bits,lut=lut)
        return lut
        
def decompress(huffman_tree,compressedfile,decompressedfile):
    #repeatedly decode and pop bits from a bitstream until the pseudo eof is encountered
    with open(decompressedfile,'w') as outfile:
        with input_bitstream(compressedfile) as bitstream:
            at_eof = False
            while True:
                token = huffman_tree.decode(bitstream)
                at_eof = (token == EOF_SYMBOL)
                #stop reading at our psuedo eof so we don't read garbage
                if at_eof:
                    break
                outfile.write(token)
            

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="pass infile to huff-compress/decompress for compression/decompression")
parser.add_argument('--bench', dest='bench', action='store_true')
parser.set_defaults(bench=False)
args= parser.parse_args()

compressedfile = args.infile
root = os.path.splitext(compressedfile)[0]
pklfile = root +'-symbol-model.pkl'
decompressedfile = root +'-decompressed.txt'

with bench_section("Decoding compressed file",args.bench):
    with open(pklfile,'rb') as pklstream:
        huffman_tree = pickle.load(pklstream)
        
    decompress(huffman_tree,compressedfile,decompressedfile)