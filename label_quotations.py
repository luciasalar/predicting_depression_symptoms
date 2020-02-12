"""This script annotate whether a post is a quote."""
#from QuoteDetector import *
import re
import csv
import spacy
import gc


class MyQuote:
    """Create a class to store each variable as an object."""

    def __init__(self, text):
        """Read in CSV, process."""
        self.quoteText = text
        self.quoteID = hash(self.quoteText)
        self.textID = []
        self.link = []
        self.name = []
        self.description = []
        self.scores = []
        self.cos_en = []
        self.cos_lg = []

    def __hash__(self):
        """Hash with quote ID."""
        return self.quoteID

    def __str__(self):
        """Hash with quote ID."""
        return "Object text: " + self.quoteText + '\n' +\
               "Links: " + str(self.link) + '\n' +\
               "Names: " + str(self.name) + '\n' +\
               "Description " + str(self.description) + '\n' +\
               "Scores: " + str(self.scores)


class GetLabels:
    """Get quotation labels."""

    def __init__(self):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.search_file = 'all_search.csv'

    def preprocess_fbp(self, sent):
        """Preprocess text."""
        sent = sent.replace('**bobsnewline**', '')
        words = str(sent).lower().split()
        new_words = []
        for w in words:
            w = re.sub(r'[0-9]+', '', w)

            new_words.append(w)
        return ' '.join(new_words)

    def preprocess(self, sent):
        """Preprocess text."""
        words = str(sent).lower().split()
        new_words = []
        for w in words:
            w = re.sub(r'[0-9]+', '', w)
            new_words.append(w)
        return ' '.join(new_words)

    def hash_results(self):
        """Hash label result as dictionary."""
        objects = {}
        with open(self.path + self.search_file, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                texthash = hash(row['text'])
                if texthash not in objects:
                    objects[texthash] = MyQuote(row['text'])
                objects[texthash].link.append(row['link'])
                objects[texthash].name.append(row['name'])
                objects[texthash].description.append(row['description'])
                objects[texthash].textID.append(row['textID'])
                line = self.preprocess(row['name'])
                # count keywords
                count = line.count("lyric") + line.count("lyrics") + line.count("quote") + line.count("quotes")
                objects[texthash].scores.append(count)

        return objects

    def cosine_sim(self):
        """For each doc, compute cosine similarity between doc and returned text
        in dictionary, store result in dict."""
        nlplg = spacy.load('en_core_web_lg')
        nlp = spacy.load('en')
        results = self.hash_results()
        for item in results:
            str1 = ''.join(results[item].description)

            # preprocess documents
            pro_doc1 = self.preprocess(str1)
            pro_doc2 = self.preprocess_fbp(results[item].quoteText)
            # use spacy model on text
            doc2 = nlp(pro_doc2)
            doc1 = nlp(pro_doc1)
            doc2lg = nlplg(pro_doc2)
            doc1lg = nlplg(pro_doc1)

            # compute cosine similarity
            results[item].cos_en.append(doc1.similarity(doc2))
            results[item].cos_lg.append(doc1lg.similarity(doc2lg))
        return results

    def get_quote_label(self, filename):
        """Here we create a class to store each variable as an object
            Return the count of key words 'lyric, lyrics, quote, quotes' in the
            name of the website, because the name of the website contains most
            of the information we need
            Count the keywords and store it as Score. Score is a vector that
            contain keyword counts in each search result compute cosine
            similarity between post and retrieve website content"""

        objects = self.cosine_sim()
        f = open(self.path + filename, 'w')
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["textID"] + ["text"] + ["count"] + ["cosineSim_en"] +
        ["cosineSim_lg"] + ['label'])
        f.close()
        for item in objects:
            f = open(self.path + filename, 'w')
            if objects[item].cos_lg[0] > 0.988:
                writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] +
                [objects[item].cos_en[0]] + [objects[item].cos_lg[0]] + ['Quote'])
            # elif (objects[item].cos_lg[0] <= 0.988 and objects[item].cos_en[0] > 0.91):
            # writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]]+  [objects[item].cos_lg[0]] + ['Quote'])

            elif (objects[item].cos_lg[0] < 0.988 and objects[item].cos_lg[0] >= 0.975) or (objects[item].cos_lg[0] <= 0.988 and objects[item].cos_en[0] > 0.91):
                # check if title has keywords
                count = 0
                for score in objects[item].scores:
                    if score > 0:# 1 is 0 before we add 1 for smoothing
                        count = count + 1
                if count > 2:
                    writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]]+  [objects[item].cos_lg[0]] + ['quote'])

            elif (objects[item].cos_lg[0] < 0.975 and objects[item].cos_lg[0] > 0.90):
                count = 0
                for score in objects[item].scores:
                    if score > 0:# 1 is 0 before we add 1 for smoothing
                        count = count + 1
                if count > 4:
                    writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]] + [objects[item].cos_lg[0]] + ['quote'])
            else:
                writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]] + [objects[item].cos_lg[0]] +['NonQuote'])
       # else:
           # writer.writerow([objects[item].quoteID] + [objects[item].quoteText] + [objects[item].scores] + ['null']+['NotQuote'])
            f.close()
            gc.collect()





# get cosine similarity score, this table is used as feature directly
l = GetLabels()
l.get_quote_label('quoteLabel1.csv')
