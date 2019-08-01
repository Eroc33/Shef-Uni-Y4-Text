import math, operator, itertools

class BinaryModel:
    """Model implementing binary metric"""
    def __init__(self,retreiver,index):
        self.binary = {}
        for term,doc_count in index.items():
            for doc,count in doc_count.items():
                self.binary.setdefault(doc,{})[term] = 1
                
    def metric(self):
        return self.binary
        
    def query_metric(self,query):
        return {term: 1 if count>=1 else 0 for term, count in query.items()}

class TfModel:
    """Model implementing tf metric"""
    def __init__(self,retreiver,index):
        self.tf = {}
        for term,doc_count in index.items():
            for doc,count in doc_count.items():
                self.tf.setdefault(doc,{})[term] = count
                
    def metric(self):
        return self.tf
        
    def query_metric(self,query):
        return query

class TfIdfModel:
    """Model implementing tf.idf metric"""
    def __init__(self,retreiver,index):
        self.tf = index
        
        self.df = {term: len(doc_count) for term, doc_count in index.items()}
        self.idf = {term: math.log( retreiver.doc_count / df ) for term, df in self.df.items()}
            
        self.tf_idf = {}
        for term,doc_count in index.items():
            for doc,count in doc_count.items():
                self.tf_idf.setdefault(doc,{})
                self.tf_idf[doc][term] = self.tf[term][doc] * self.idf[term]
                
    def metric(self):
        return self.tf_idf
        
    def query_metric(self,query):
        query_tf_idf = {}
        for term, tf in query.items():
            #here we get with a default of 0 (though the idf for a term which does
            #not appear in the document set is undefined) as we multiply it by 0
            #in the dot product (as the dimension for this term must be 0 in every
            #document if it doesn't appear in any document) and so the value only matters
            #as far as the length of the query is concerned
            query_tf_idf[term] = tf * self.idf.get(term,0)
        return query_tf_idf

class Retrieve:
    """
        Retrieve class.
        Delegates to one of the model classes to calculate the metric that
        similarity scores are calculated on.
    """
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        
        doc_set = set()
        for term,doc_count in index.items():
            doc_set = doc_set.union(set(doc_count))
        self.doc_count = len(doc_set)
          
        if termWeighting == "binary":
            self.model = BinaryModel(self,index)
        elif termWeighting == "tf":
            self.model = TfModel(self,index)
        elif termWeighting == "tfidf":
            self.model = TfIdfModel(self,index)
        else:
            raise Exception("Unknown term weighting scheme: %s"%(termWeighting))
                
        self.doc_sizes = {doc:euclidean_size(vector) for doc, vector in self.model.metric().items()}
        
    # Method performing retrieval for specified query
    def forQuery(self, query):
        query_vector = self.model.query_metric(query)
        query_size = euclidean_size(query_vector)
            
        scores = {doc: similarity(query_vector,self.model.metric()[doc],sizes=[query_size, doc_size]) for doc,doc_size in self.doc_sizes.items()}
        ranked = sorted(scores,key=scores.get,reverse=True)
        
        #ignoring results below a certain score seems to work better
        #than ignoring results below a certain rank
        found_docs = list(itertools.takewhile(lambda rank: scores.get(rank) > 0.1,ranked))
        
        return found_docs
        
def euclidean_size(vector):
    return math.sqrt(
        sum( (weight ** 2) for weight in vector.values() )
    )
    
def similarity(a,b,sizes=[None,None]):
    """
        Cosine similarity.
        pass sizes if lengths are already known to prevent them being
        recalculated. you may set unknown sizes to None for them to be
        calculated
    """
    #calculate sizes if needed
    sizes = [ size if size is not None else euclidean_size(vec) for size,vec in zip(sizes,[a,b]) ]
    
    #we only need to calculate the dot for terms that are shared between the
    #   query and a document, as other terms are implicitly zero so do not
    #   contribute to the dot product
    shared_terms = set(a) & set(b)
    
    dot = 0
    for term in shared_terms:
        dot += a[term] * b[term]
        
    return dot/(sizes[0]*sizes[1])