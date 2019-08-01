import argparse
import re
import os
import pickle
import heapq
from contextlib import contextmanager
from collections import defaultdict
import time

#psuedo eof symbol
#an integer cannot actually occur as a token (they are always a string), so it
#should never be confused for a real token
EOF_SYMBOL = -1

class OutputBitstream:
    """Class to wrap an io object abstracting bitwise operations in the writing of bits"""
    def __init__(self,file):
        self.file = file
        self.buffer = []
        
    def write(self,bits):
        self.buffer.extend(bits)
        self.flush()
            
    def flush(self):
        byte_list = []
        while len(self.buffer) >= 8:
            byte_bits = self.buffer[:8]
            self.buffer = self.buffer[8:]
            byte = 0
            for bit in byte_bits:
                byte = (byte << 1 | bit)
            byte_list.append(byte)
        self.file.write(bytes(byte_list))
    
    def write_eof(self,bits):
        self.buffer.extend(bits)
        padding_required = len(self.buffer) % 8
        if padding_required != 0:
            self.buffer.extend([0]*padding_required)
        self.flush()

@contextmanager
def output_bitstream(file_name):
    with open(file_name,'wb') as file:
        output_bitstream = OutputBitstream(file)
        try:
            yield output_bitstream
        finally:
            output_bitstream.flush()

@contextmanager            
def bench_section(section_name,active):
    if active:
        start = time.perf_counter()
    yield
    if active:
        end = time.perf_counter()
        print(f'Section {section_name} took {end-start} seconds')

#Note that storing the ordering for the headp outside of the nodes lets us decrease pickle file size
class OrderBy:
    """Used to store probabilities of nodes outside of HNode so they can be removed when pickling"""
    def __init__(self,value,p):
        self.value = value
        self.p = p

    #override comparison operators to allow use in heapq
    def __lt__(self, other):
        return self.p < other.p

    def ___le__(self, other):
        return self.p <= other.p

    def __eq__(self, other):
        return self.p == other.p

    def __ne__(self, other):
        return self.p != other.p

    def __gt__(self, other):
        return self.p > other.p

    def __ge__(self, other):
        return self.p >= other.p

#Note that splitting Hnode and HNodeLeaf both slightly decreases logic complexity, and decreases pickle file size
class HNodeLeaf:
    def __init__(self,value=None):
        self.value = value
        
    def decode(self,bitstream):
        return self.value
                
    def make_lut(self,bits=[],lut={}):
        """Turns a huffman tree into a dict based lut which should be more efficient for compression"""
        lut[self.value] = bits
        return lut
        
#if using python >=3.7 this should probably be a dataclass
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

def word_tokenize(str):
    offset = 0
    word_re = re.compile('[a-zA-Z]+')
    tokens = []
    remaining = len(str)
    while offset < len(str):
        word = word_re.search(str,offset)
        if word:
            word_start = word.span()[0]
            word_end = word.span()[1]
            pre_word = str[offset:word_start]
            offset = word_end
            tokens.extend(list(pre_word))
            tokens.append(word.group())
        else:
            #case of dangling non-word tokens
            tokens.extend(list(str[offset:]))
            offset = len(str)
        
    return tokens
    
def tokenize(infile,symbolmodel):
    with open (infile, "r") as file:
        input=file.read()

    if symbolmodel == "word":
        tokens = word_tokenize(input)
    else:
        tokens = list(input)
    #add our psuedo eof symbol
    tokens.append(EOF_SYMBOL)
    return tokens
    
def calc_probabilites(tokens):
    total_token_count = len(tokens)
    unique_tokens = defaultdict(int)
    for token in tokens:
        unique_tokens[token] += 1
    probs = [OrderBy(HNodeLeaf(value=token),count/total_token_count) for token,count in unique_tokens.items()]
    return probs
    
def build_tree(token_probs):
    heap = token_probs
    heapq.heapify(heap)
    #while we have more than one node remaining combine them unde one parent
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        heapq.heappush(heap,OrderBy(HNode(left=l.value,right=r.value),l.p+r.p))
    return heap[0].value
    
def compress(tokens,huffman_tree,outfile):
    #lookup each token in the input and write its bit value to the output
    with open(outfile,'wb') as outstream:
        bitstream = OutputBitstream(outstream)
        lut = huffman_tree.make_lut()
        for token in tokens:
            if token == EOF_SYMBOL:
                bitstream.write_eof(lut[token])
            else:
                bitstream.write(lut[token])

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="pass infile to huff-compress/decompress for compression/decompression")
parser.add_argument("-s", "--symbolmodel", help="specify character- or word-based Huffman encoding -- default is character",
                    choices=["char","word"])
parser.add_argument('--bench', dest='bench', action='store_true')
parser.set_defaults(bench=False)
args= parser.parse_args()

infile = args.infile
root = os.path.splitext(infile)[0]
pklfile = root +'-symbol-model.pkl'
compressedfile = root +'.bin'

with bench_section("Building symbol model",args.bench):
    tokens = tokenize(infile,args.symbolmodel)
    token_probs = calc_probabilites(tokens)
    huffman_tree = build_tree(token_probs)
with bench_section("Encode input file",args.bench):
    compress(tokens,huffman_tree,compressedfile)
    
with open(pklfile,'wb') as pklstream:
    pickle.dump(huffman_tree,pklstream,protocol=pickle.HIGHEST_PROTOCOL)
    
if args.bench:
    compressed_file_size = os.stat(compressedfile).st_size
    print(f'Compressed file is {compressed_file_size} bytes')
    symbol_file_size = os.stat(pklfile).st_size
    print(f'Symbol file is {symbol_file_size} bytes')