"""

@author : Eric Yamga

Objective of this script is to parse the CODA19 database drugs and to turn them into meaningful classes.
The output of this script is a CSV file that serves as a dictionary between drug names and classes.
The classes are those found in the MESH database.
This vocabulary was chosen as their drug classes provided the best tradeoff of generalization and precision.

We also use the RxClass API from RxNorm in the first portion of the script to facilitate the translation.
"""


import requests
import json
from pprint import pprint as pp
from functools import reduce
import pandas as pd

# The class was modified as such as to retrieve MESH drug class
class Rx_APIWrapper(object):

    def __init__(self):
        self.base_uri_interactions = 'https://rxnav.nlm.nih.gov/REST/interaction'
        self.base_uri_class = 'https://rxnav.nlm.nih.gov/REST/rxclass'
        self.base_uri_norm = 'https://rxnav.nlm.nih.gov/REST'
        self.drug_class_soaps = []

    def make_request(api_url_getter):
        def wrapper(self, *args):
            return requests.get(api_url_getter(self, *args)).json()
        return wrapper

    def sanitize(self, opts):
        if type(opts) is dict:
            return reduce(lambda acc, i: acc + "&{}={}".format(i[0], i[1]), opts.items(), "")
        return ''

    @make_request
    def get_interaction_uri(self):
        return self.base_uri_class + '/interaction.json?rxcui=' + str(341248)

    @make_request
    def find_class_by_id(self, drug_class_id):
        return self.base_uri_class + "/class/byId.json?classId={}".format(drug_class_id)

    @make_request
    def find_class_by_name(self, name):
        return self.base_uri_class + "/class/byName.json?className={}".format(name)

    @make_request
    def find_class_by_cui(self, name, opts=None):
        return self.base_uri_class + "/class/byRxcui.json?rxcui={}&relaSource=SNOMEDCT&relas=isa_disposition".format(name) + self.sanitize(opts)

    @make_request
    def find_class_by_drug_name(self, drug_name, opts=None):
        return self.base_uri_class + "/class/byDrugName.json?drugName={}&relaSource=MESH".format(
            drug_name) + self.sanitize(opts)

    @make_request
    def find_class_by_drug_name_sno(self, drug_name, opts=None):
        return self.base_uri_class + "/class/byDrugName.json?drugName={}&relaSource=SNOMEDCT&relas=isa_disposition".format(drug_name) + self.sanitize(opts)

    ###
    @make_request
    def find_similar_classes_by_class(self, class_id, opts=None):
        return self.base_uri_class + "/class/similar.json?classId={}".format(class_id) + self.sanitize(opts)

    @make_request
    def find_similar_classes_by_drug_list(self, drug_ids, opts=None):
        return self.base_uri_class + "/class/similarByRxcuis?rxcuis={}".format(drug_ids) + self.sanitize(opts)

    @make_request
    def get_all_classes(self, class_types=None):
        return self.base_uri_class + "/allClasses.json" + self.sanitize(class_types)

    @make_request
    def get_class_contexts(self, class_id):
        return self.base_uri_class + "/classContext.json?classId={}".format(class_id)

    @make_request
    def get_class_graph(self, class_id):
        return self.base_uri_class + "/classGraph.json?classId={}".format(class_id)

    @make_request
    def get_class_members(self, class_id, opts=None):
        return self.base_uri_class + "/classMembers.json?classId={}".format(class_id) + self.sanitize(opts)

    @make_request
    def get_class_tree(self, class_id):
        return self.base_uri_class + "/classTree.json?classId={}".format(class_id)

    @make_request
    def get_class_types(self):
        return self.base_uri_class + "/classTypes.json"

    @make_request
    def get_relationships(self, rela_source):
        return self.base_uri_class + "/relas.json?relaSource={}".format(rela_source)

    @make_request
    def compare_classes(self, class_id_1, opts=None):
        return self.base_uri_class + "/similarInfo.json?" + self.sanitize(opts)[1:]

    @make_request
    def get_sources_of_drug_class_relations(self):
        return self.base_uri_class + "/relaSources.json"

    @make_request
    def get_spelling_suggestions(self, term, type_of_name):
        return self.base_uri_class + "/spellingsuggestions.json?term={}&type={}".format(term, type_of_name)

    def save(self):
        pp(json.dumps(self.req.json(), "/req_1.json"))


# Creating an instance of the RxAPIWrapper
f = Rx_APIWrapper()
# Testing API different sources
# f.find_class_by_drug_name('bisoprolol')['rxclassDrugInfoList']['rxclassDrugInfo'][0]['rxclassMinConceptItem']['className']


# Loading all drug names of interest
# This CSV contains all the drugs found in the CODA19 database during the entire stay of all patients
drugs = pd.read_csv('/data8/projets/Mila_covid19/code/eyamga/phenotyper/code/py_eyamga/csv/covid_drugs.csv')
# We only keep the unique drugs as we are trying to create a dictionary from drug names to classes
drugs_list_original = pd.Series(drugs['drug_name'].unique())
# drugs_list_mod = drugs_list_original.str.replace('ine', 'in', regex=False)

# Find class for all drugs in covid db
drug_class_dict = {}
for i in drugs_list_original:
    if 'rxclassDrugInfoList' in f.find_class_by_drug_name(i):
        #if cui available, could use the find_class_by_cui function
        drug_class_dict[i] = f.find_class_by_drug_name(i)['rxclassDrugInfoList']['rxclassDrugInfo'][0]['rxclassMinConceptItem']['className']
    else:
        drug_class_dict[i] = 0


# Manually optimizing the dictionary :
'''' Script to replace values without replacing keys, but was not used ultimately
for key, value in drug_class_dict.items():
    # do something with value
    inputdict[key] = newvalue
'''

def manualcorrection():
    drug_class_dict['acetaminophene']='Acetaminophene'
    drug_class_dict['aspirin']='Platelet Aggregation Inhibitors'

    drug_class_dict['methylprednisolone']='Glucocorticoids'
    drug_class_dict['prednisone']='Glucocorticoids'
    drug_class_dict['dexamethasone']='Glucocorticoids'
    drug_class_dict['hydrocortisone'] = 'Glucocorticoids'
    drug_class_dict['prednisolone'] = 'Glucocorticoids'

    drug_class_dict['amoxicilline/clavulanate']='Anti-Bacterial Agents'
    drug_class_dict['piperacilline/tazobactam'] = 'Anti-Bacterial Agents'
    drug_class_dict['metronidazole'] = 'Anti-Bacterial Agents'
    drug_class_dict['co-trimoxazole'] = 'Anti-Bacterial Agents'
    drug_class_dict['fidaxomicine'] = 'Anti-Bacterial Agents'
    drug_class_dict['moxifloxacine'] = 'Anti-Bacterial Agents'
    drug_class_dict['penicilline g sodique'] = 'Anti-Bacterial Agents'
    drug_class_dict['nitrofurantoine'] = 'Anti-Bacterial Agents'
    drug_class_dict['levofloxacine'] = 'Anti-Bacterial Agents'
    drug_class_dict['gentamicine'] = 'Anti-Bacterial Agents'
    drug_class_dict['nitrofurantoine'] = 'Anti-Bacterial Agents'


    drug_class_dict['caspofongine'] = 'Antifungal Agents'


    drug_class_dict['risedronate']='Bone Density Conservation Agents'
    drug_class_dict['sevelamer chlorhydrate']='Bone Density Conservation Agents'
    drug_class_dict['zoledronate'] = 'Bone Density Conservation Agents'
    drug_class_dict['lithium carbonate'] = 'Bone Density Conservation Agents'



    drug_class_dict['glycerine']='Laxatives'
    drug_class_dict['polyethylene glycol'] = 'Laxatives'
    drug_class_dict['sennosides a-b'] = 'Laxatives'
    drug_class_dict['lactulose'] = 'Laxatives'
    drug_class_dict['bisacodyl'] = 'Laxatives'
    drug_class_dict['docusate sodique'] = 'Laxatives'


    drug_class_dict['calcium carbonate'] = 'Vitamins'
    drug_class_dict['folique acide'] = 'Vitamins'
    drug_class_dict['phosphate de potassium'] = 'Vitamins'
    drug_class_dict['phosphate acide de sodium'] = 'Vitamins'
    drug_class_dict['potassium chlorure'] = 'Vitamins'
    drug_class_dict['vitamin b1'] = 'Vitamins'
    drug_class_dict['potassium chlorure'] = 'Vitamins'
    drug_class_dict['vitamin c'] = 'Vitamins'
    drug_class_dict['vitamin b6'] = 'Vitamins'
    drug_class_dict['vitamin e'] = 'Vitamins'
    drug_class_dict['vitamine a'] = 'Vitamins'
    drug_class_dict['bicarbonate sodium'] = 'Bicarbonate'


    drug_class_dict['risperidone']='Antipsychotic Agents'
    drug_class_dict['quetiapine']='Antipsychotic Agents'
    drug_class_dict['aripiprazole']='Antipsychotic Agents'
    drug_class_dict['olanzapine'] = 'Antipsychotic Agents'
    drug_class_dict['clozapine'] = 'Antipsychotic Agents'



    drug_class_dict['venlafaxine']='Antidepressive Agents'
    drug_class_dict['escitalopram']='Antidepressive Agents'
    drug_class_dict['amitriptyline']='Antidepressive Agents'
    drug_class_dict['duloxetine'] = 'Antidepressive Agents'
    drug_class_dict['paroxetine'] = 'Antidepressive Agents'
    drug_class_dict['fluoxetine'] = 'Antidepressive Agents'
    drug_class_dict['paroxetine'] = 'Antidepressive Agents'
    drug_class_dict['valproique acide'] = 'Antidepressive Agents'


    drug_class_dict['levothyroxine']='Levothyroxine'
    drug_class_dict['lorazepam']='Benzodiazepines'
    drug_class_dict['oxazepam'] = 'Benzodiazepines'
    drug_class_dict['midazolam'] = 'Benzodiazepines'
    drug_class_dict['diazepam'] = 'Benzodiazepines'
    drug_class_dict['clonazepam'] = 'Benzodiazepines'
    drug_class_dict['alprazolam'] = 'Benzodiazepines'
    drug_class_dict['diazepam'] = 'Benzodiazepines'
    drug_class_dict['clonazepam'] = 'Benzodiazepines'
    drug_class_dict['zopiclone'] = 'Benzodiazepines'
    drug_class_dict['zopiclone'] = 'Benzodiazepines'

    drug_class_dict['propofol'] = 'Sedation'
    drug_class_dict['dexmedetomidine'] = 'Sedation'

    drug_class_dict['humulin r']='Hypoglycemic Agents'
    drug_class_dict['novolin ge nph'] = 'Hypoglycemic Agents'
    drug_class_dict['insulin degludec'] = 'Hypoglycemic Agents'
    drug_class_dict['semaglutide'] = 'Hypoglycemic Agents'
    drug_class_dict['saxagliptine'] = 'Hypoglycemic Agents'
    drug_class_dict['novolin ge 30/70'] = 'Hypoglycemic Agents'
    drug_class_dict['novolin ge 50/50'] = 'Hypoglycemic Agents'
    drug_class_dict['novolin ge toronto'] = 'Hypoglycemic Agents'
    drug_class_dict['dapagliflozine'] = 'Hypoglycemic Agents'
    drug_class_dict['empagliflozine'] = 'Hypoglycemic Agents'
    drug_class_dict['linagliptine'] = 'Hypoglycemic Agents'
    drug_class_dict['empagliflozine'] = 'Hypoglycemic Agents'

    drug_class_dict['ipratropium/salbutamol']='Bronchodilator Agents'
    drug_class_dict['glycopyrronium bromide'] = 'Bronchodilator Agents'
    drug_class_dict['glycopyrronium+indacaterol'] = 'Bronchodilator Agents'
    drug_class_dict['indacaterol'] = 'Bronchodilator Agents'
    drug_class_dict['formoterol/budesonide'] = 'Bronchodilator Agents'
    drug_class_dict['indacaterol'] = 'Bronchodilator Agents'
    drug_class_dict['umeclidinium/vilanterol'] = 'Bronchodilator Agents'
    drug_class_dict['salmeterol/fluticasone'] = 'Bronchodilator Agents'
    drug_class_dict['formoterol/mometasone'] = 'Bronchodilator Agents'
    drug_class_dict['aclidinium'] = 'Bronchodilator Agents'
    drug_class_dict['umeclidinium'] = 'Bronchodilator Agents'
    drug_class_dict['vilanterol et fluticasone'] = 'Bronchodilator Agents'
    drug_class_dict['umeclidinium'] = 'Bronchodilator Agents'

    drug_class_dict['spironolactone']='Diuretics'
    drug_class_dict['acetazolamide'] = 'Diuretics'
    drug_class_dict['metolazone'] = 'Diuretics'

# DOACs are in there own category as AntiXa
    drug_class_dict['dalteparine']='Anticoagulants'
    drug_class_dict['danaparoide'] = 'Anticoagulants'
    drug_class_dict['warfarine'] = 'Anticoagulants'
    drug_class_dict['argatroban'] = 'Anticoagulants'
    drug_class_dict['fondaparinux'] = 'Anticoagulants'


    drug_class_dict['labetalol']='Antihypertensive Agents'
    drug_class_dict['ramipril'] = 'Antihypertensive Agents'
    drug_class_dict['labetalol'] = 'Antihypertensive Agents'
    drug_class_dict['telmisartan'] = 'Antihypertensive Agents'
    drug_class_dict['captopril'] = 'Antihypertensive Agents'
    drug_class_dict['clonidine'] = 'Antihypertensive Agents'
    drug_class_dict['enalapril'] = 'Antihypertensive Agents'
    drug_class_dict['valsartan'] = 'Antihypertensive Agents'
    drug_class_dict['losartan'] = 'Antihypertensive Agents'
    drug_class_dict['trandolapril'] = 'Antihypertensive Agents'
    drug_class_dict['atenolol'] = 'Adrenergic beta-Antagonists'

    drug_class_dict['norepinephrine'] ='Vasopressors'
    drug_class_dict['midodrine'] = 'Vasopressors'
    drug_class_dict['vasopressine'] = 'Vasopressors'
    drug_class_dict['dobutamine'] = 'Vasopressors'
    drug_class_dict['dopamine'] = 'Vasopressors'
    drug_class_dict['epinephrine'] = 'Vasopressors'
    drug_class_dict['milrinone'] = 'Vasopressors'

    drug_class_dict['rosuvastatine'] = 'Anticholesteremic Agents'

    drug_class_dict['mycophenolate mofetil'] = 'Immunosuppressive Agents'
    drug_class_dict['mycophenolate sodium'] = 'Immunosuppressive Agents'
    drug_class_dict['methotrexate'] = 'Immunosuppressive Agents'
    drug_class_dict['azathioprine'] = 'Immunosuppressive Agents'


    drug_class_dict['naproxene'] = 'Anti-Inflammatory Agents, Non-Steroidal'

    drug_class_dict['cyclobenzaprine'] = 'Analgesics'

    drug_class_dict['darunavir'] = 'HIV medication'
    drug_class_dict['dolutegravir'] = 'HIV medication'
    drug_class_dict['etravirine'] = 'HIV medication'
    drug_class_dict['ritonavir'] = 'HIV medication'
    drug_class_dict['abacavir'] = 'HIV medication'
    drug_class_dict['lamivudine'] = 'HIV medication'
    drug_class_dict['tenofovir disoproxil'] = 'HIV medication'
    drug_class_dict['biktarvy'] = 'HIV medication'
    drug_class_dict['doravirine'] = 'HIV medication'
    drug_class_dict['lopinavir+ritonavir'] = 'HIV medication'

    drug_class_dict['isoniazide'] = 'Antitubercular Agents'
    drug_class_dict['rifampine'] = 'Antitubercular Agents'
    drug_class_dict['entecavir'] = 'Antitubercular Agents'
    drug_class_dict['trazodone']='Vasopressors'

    drug_class_dict['levodopa/benserazide']='Antiparkinson Agents'
    drug_class_dict['pramipexole'] = 'Antiparkinson Agents'
    drug_class_dict['levodopa/carbidopa/entacapone 150/37.5/200 mg'] = 'Antiparkinson Agents'
    drug_class_dict['stalevo 125/31,25/200 mg'] = 'Antiparkinson Agents'
    drug_class_dict['levodopa/carbidopa'] = 'Antiparkinson Agents'
    drug_class_dict['stalevo 125/31,25/200 mg'] = 'Antiparkinson Agents'

    drug_class_dict['rocuronium'] = ['Neuromuscular Blocking Agents']

    drug_class_dict['nystatine']=0
    drug_class_dict['ciclesonide'] =0
    drug_class_dict['brimonidine'] = 0
    drug_class_dict['dorzolamide'] = 0
    drug_class_dict['chlorhexidine'] = 0
    drug_class_dict['mupirocine'] = 0
    drug_class_dict['atropine sulfate'] = 0
    drug_class_dict['tizanidine'] = 0
    drug_class_dict['beclomethasone'] = 0
    drug_class_dict['orphenadrine citrate'] = 0
    drug_class_dict['zoplicone'] = 0

    return drug_class_dict

manualcorrection()

drug_class_df = pd.DataFrame(drug_class_dict.items())
drug_class_df.columns = ['drug_name', 'drug_class']
length = len(drug_class_df.iloc[:,1].unique())

drug_class_df['drug_name'=='propofol',]
drug_class_df.to_csv('/data8/projets/Mila_covid19/code/eyamga/phenotyper/code/py_eyamga/csv/drug_class_dict.csv')

