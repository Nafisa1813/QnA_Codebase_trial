from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import print_documents
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import TfidfRetriever
from haystack.nodes import TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
import pandas as pd
from IPython.display import display, clear_output
from IPython.core.display import HTML
from IPython.display import HTML
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import pprint
import numpy as np
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
import json
from scipy.spatial import distance
import openai
import configparser
from string import ascii_lowercase as alc
import pprint
import ipywidgets as widgets
from ipywidgets import Layout
import traceback
from haystack.utils import print_documents
from nltk.tokenize import sent_tokenize
from haystack import Pipeline, Document
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter(action='ignore', category=ResourceWarning)

# from  transformers  import  AutoTokenizer, AutoModelWithLMHead, pipeline
# import torch
# from transformers import AutoConfig, AutoModelForQuestionAnswering
readerMem = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa") 


Patient_query = [
        '-Select-',
        'How many in-person visits would be required?',
        'What will be the duration of study?',
        'What are the side effects of the treatment?',
        'Will the patient be able to continue their current medications and treatments during the study?',
        'I have diabetes can I participate in the trial?',
        'What happens if the patient experiences adverse events during the study?',
        'I am 81 year old Male, can I participate in the trial?'
        
    ]

Site_query = [
    '-Select-',
    'What study assessments are followed in the trial?',
    'What are the storage conditions for Apibaxin?',
    'What measures are in place to protect participant confidentiality and privacy?',
    'What is the process of reporting adverse events?',
    'In what cases will the trial be terminated?'
    
]


class Paragraph_Retrieval:
    def __init__(self, document_data, user_input):
        self.document_data = document_data
        self.load_model()
        self.query_embedding = self.get_embedding(user_input.strip())

    def load_model(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.model.max_seq_length = 512

    def process_document_data(self):

        data_dict = dict(zip(self.document_data.Section_Name, self.document_data.Content))
        return data_dict


    def get_embedding(self, text):
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding

    def similartiy_of_each_para(self, processed_para):

#         similarity_score_of_each_para = []
#         for i in processed_para:
#             if i != '':
#                 emb = self.get_embedding(i.strip())

#                 cosine_sim = 1 - distance.cosine(self.query_embedding, emb)
#                 similarity_score_of_each_para.append((i, cosine_sim))
#         return similarity_score_of_each_para
        if len(processed_para)!=0:
            emb = self.get_embedding(processed_para.strip())
            cosine_sim = 1 - distance.cosine(self.query_embedding, emb)
            return cosine_sim
#             print('First - ', cosine_sim)
            
        else:
            cosine_sim = 0        
            return cosine_sim

    def run_paragraph_retrieval(self):
  
        self.document_data['Cosine_score'] = [0]*len(self.document_data)
        self.document_data['Cosine_score'] = self.document_data['Content'].apply(lambda x:self.similartiy_of_each_para(x))
        final_df = self.document_data.sort_values(by=['Cosine_score'], ascending=False)
        final_df.reset_index(drop=True, inplace=True)
        return final_df

    

class PrepareData:
    def __init__(self):
#         self.protocol_obj = ProtocolDataIngestion()
        pass
    
    def protocolIngestion(self, protocol_number):
        if protocol_number=='001':
            preprocessed_df = pd.read_csv('Protocol001.csv')
            preprocessed_df = preprocessed_df.fillna('')
        else:
            preprocessed_df = pd.read_csv('Protocol000.csv')
            preprocessed_df = preprocessed_df.fillna('')
        return preprocessed_df
    
    def utilsPrepareData(self, protocol_number, preprocessed_df):
        
        if protocol_number=='001':
            to_remove_sections = ['APPENDIX 1 INTERNATIONAL STAGING SYSTEM',
                          'APPENDIX 3 PREPARATION AND ADMINISTRATION OF ELOTUZUMAB',
                          'Figure 3.1-1: Study Design Schematic',                   
                         'LIST OF ABBREVIATIONS',
                          'TABLE OF CONTENTS',
                         'Table 4-1: Study Drugs for CA204116 Treatment Period',
                         'Table 4.5.1-1: Treatment Schedule',
                         'Table 4.5.1.2-1: Corticosteroid Premedication',
                         'Table 4.5.4.2-1: Dexamethasone Dose Reductions',
                         'Table 4.5.4.2-2: Dexamethasone Dose Levels',
                         'Table 4.5.4.3-1: Treating Thrombocytopenia Related to Lenalidomide',
                         'Table 4.5.4.3-2: Treating Neutropenia Related to Lenalidomide',
                         'Table 4.5.4.3-3: Lenalidomide Dose Adjustments in Subjects with Renal Impairment',
                         'Table 5.1-1: Screening Procedural Outline (CA204116)',
                         'Table 5.1-2: Short-term Procedural Outline (CA204116) Cycles 1 &',
                         'Table 5.1-3: Long-term Procedural Outline (CA204116) Cycles 3 and Beyond',
                         'Table 5.4.4-1: Safety Laboratory Assessments (may be drawn up to three days prior visit)',
                         'Table 5.5.2-1: Bone marrow samples',
                         'Table 5.5.4-1: IMWG Criteria for Response',
                         'Table 5.5.4-2: IMWG Criteria for Progression',
                         'Table 5.6-1: PK and ADA Sampling Schedule',
                         ]
        else:
            to_remove_sections = [
                'TABLE OF CONTENTS',
                 'Table 4-1: BMS Supplied Study Drugs for CV185316',
                 'Table 5.1-1: Baseline and Randomization (CV185316)',
                 'Table 5.1-2: Short-term Procedural Outline (CV185316)',
                 'Table 8.1-1: Sample Size Adjustment for Non-inferiority and Superiority Test on the Primary Secondary Endpoints'
                 'Figure 3.1-1: Study Design Schematic'
                  'LIST OF ABBREVIATIONS',
                  'Figure 3.1-1: Study Design Schematic']

        required_sections = list(set(list(preprocessed_df.Section_Name)) - set(to_remove_sections))
        main_df = pd.DataFrame()
        for section in required_sections:
            main_df = main_df.append(preprocessed_df[preprocessed_df['Section_Name']==section])
        main_df.reset_index(drop=True, inplace=True)

        return main_df
    
    
    def prepareData(self, protocol_number, preprocessed_df):
    
        if protocol_number=='001':
            to_remove_sections = ['APPENDIX 1 INTERNATIONAL STAGING SYSTEM',
                          'APPENDIX 3 PREPARATION AND ADMINISTRATION OF ELOTUZUMAB',
                          'Figure 3.1-1: Study Design Schematic',                   
                         'LIST OF ABBREVIATIONS',
                          'TABLE OF CONTENTS',
                         'Table 4-1: Study Drugs for CA204116 Treatment Period',
                         'Table 4.5.1-1: Treatment Schedule',
                         'Table 4.5.1.2-1: Corticosteroid Premedication',
                         'Table 4.5.4.2-1: Dexamethasone Dose Reductions',
                         'Table 4.5.4.2-2: Dexamethasone Dose Levels',
                         'Table 4.5.4.3-1: Treating Thrombocytopenia Related to Lenalidomide',
                         'Table 4.5.4.3-2: Treating Neutropenia Related to Lenalidomide',
                         'Table 4.5.4.3-3: Lenalidomide Dose Adjustments in Subjects with Renal Impairment',
                         'Table 5.1-1: Screening Procedural Outline (CA204116)',
                         'Table 5.1-2: Short-term Procedural Outline (CA204116) Cycles 1 &',
                         'Table 5.1-3: Long-term Procedural Outline (CA204116) Cycles 3 and Beyond',
                         'Table 5.4.4-1: Safety Laboratory Assessments (may be drawn up to three days prior visit)',
                         'Table 5.5.2-1: Bone marrow samples',
                         'Table 5.5.4-1: IMWG Criteria for Response',
                         'Table 5.5.4-2: IMWG Criteria for Progression',
                         'Table 5.6-1: PK and ADA Sampling Schedule',
                         ]
        else:
            to_remove_sections = [
                'TABLE OF CONTENTS',
                 'Table 4-1: BMS Supplied Study Drugs for CV185316',
                 'Table 5.1-1: Baseline and Randomization (CV185316)',
                 'Table 5.1-2: Short-term Procedural Outline (CV185316)',
                 'Table 8.1-1: Sample Size Adjustment for Non-inferiority and Superiority Test on the Primary Secondary Endpoints'
                 'Figure 3.1-1: Study Design Schematic'
                  'LIST OF ABBREVIATIONS',
                  'Figure 3.1-1: Study Design Schematic']

        required_sections = list(set(list(preprocessed_df.Section_Name)) - set(to_remove_sections))
        main_df = pd.DataFrame()
        for section in required_sections:
            main_df = main_df.append(preprocessed_df[preprocessed_df['Section_Name']==section])
        main_df.reset_index(drop=True, inplace=True)


        #for haystack
        count = 0
        all_doc_dict = [] 
        for index, rows in main_df.iterrows():         
            temp_dict = {} 
            meta_dict = {}
            temp_dict['content'] = rows['Content']
            meta_dict['name'] = 'Document_'+str(count)
            meta_dict['Therapy_Area'] = 'Oncology'
    #         count = count+1
            temp_dict['meta'] = meta_dict
            temp_dict['name'] = rows['Section_Name']+'_'+'Document_'+str(count)
            count = count+1

            all_doc_dict.append(temp_dict)

        return all_doc_dict
    
    def preprocessData(self, protocol_number, preprocessed_df):
        if protocol_number=='001':
            to_remove_sections = ['APPENDIX 1 INTERNATIONAL STAGING SYSTEM',
                          'APPENDIX 3 PREPARATION AND ADMINISTRATION OF ELOTUZUMAB',
                          'Figure 3.1-1: Study Design Schematic',                   
                         'LIST OF ABBREVIATIONS',
                          'TABLE OF CONTENTS',
                         'Table 4-1: Study Drugs for CA204116 Treatment Period',
                         'Table 4.5.1-1: Treatment Schedule',
                         'Table 4.5.1.2-1: Corticosteroid Premedication',
                         'Table 4.5.4.2-1: Dexamethasone Dose Reductions',
                         'Table 4.5.4.2-2: Dexamethasone Dose Levels',
                         'Table 4.5.4.3-1: Treating Thrombocytopenia Related to Lenalidomide',
                         'Table 4.5.4.3-2: Treating Neutropenia Related to Lenalidomide',
                         'Table 4.5.4.3-3: Lenalidomide Dose Adjustments in Subjects with Renal Impairment',
                         'Table 5.1-1: Screening Procedural Outline (CA204116)',
                         'Table 5.1-2: Short-term Procedural Outline (CA204116) Cycles 1 &',
                         'Table 5.1-3: Long-term Procedural Outline (CA204116) Cycles 3 and Beyond',
                         'Table 5.4.4-1: Safety Laboratory Assessments (may be drawn up to three days prior visit)',
                         'Table 5.5.2-1: Bone marrow samples',
                         'Table 5.5.4-1: IMWG Criteria for Response',
                         'Table 5.5.4-2: IMWG Criteria for Progression',
                         'Table 5.6-1: PK and ADA Sampling Schedule',
                         ]
        else:
            to_remove_sections = [
                'TABLE OF CONTENTS',
                 'Table 4-1: BMS Supplied Study Drugs for CV185316',
                 'Table 5.1-1: Baseline and Randomization (CV185316)',
                 'Table 5.1-2: Short-term Procedural Outline (CV185316)',
                 'Table 8.1-1: Sample Size Adjustment for Non-inferiority and Superiority Test on the Primary Secondary Endpoints'
                 'Figure 3.1-1: Study Design Schematic'
                  'LIST OF ABBREVIATIONS',
                  'Figure 3.1-1: Study Design Schematic']

        required_sections = list(set(list(preprocessed_df.Section_Name)) - set(to_remove_sections))
        main_df = pd.DataFrame()
        for section in required_sections:
            main_df = main_df.append(preprocessed_df[preprocessed_df['Section_Name']==section])
        main_df.reset_index(drop=True, inplace=True)
#         main_df = main_df.groupby('Section_Name')['Content'].agg(' '.join).reset_index()
#         main_df['Content'] = main_df['Content'].apply(lambda x : [x])
        return main_df
    
    

class Reasoning:
    def __init__(self, assertion, data, num_paras):

        self.assertion = assertion
        self.data = data
       
        self.num_paras = num_paras
        self.final_output = {}
    
    def ready_paras(self):

        self.para_list = []
        context = "Protocol: \n"
        
        for i in range(self.num_paras):
            try:
                if len(context)/4 < 3000:
                    if len(self.data['Content'][i])!=0 or self.data['Content'][i]!= ' ':
                        context += '\npara-' + str(i) + ': ' + self.data['Content'][i]
                        self.para_list.append([self.data['Section_Name'][i], self.data['Content'][i], self.data['Cosine_score'][i]])
            except:
                continue
        self.ranked_paras = self.para_list
        self.retrieved_paragraphs = context
    
    def reason_w_GPT3(self):

        self.ready_paras()
        curr_assertion = 'Refer to the Protocol and provide an answer to the above question not exceeding 2 lines'

        gpt3_prompt = self.retrieved_paragraphs + '\n' + 'Question: ' + self.assertion + '\n' + curr_assertion
#         print('total input tokens', len(gpt3_prompt)/4)
        
        openai.api_key = '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'

        response1 = openai.Completion.create(
          model="text-davinci-003",
          prompt=gpt3_prompt,
          temperature=0,
          max_tokens=900,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
#         print(response1['choices'][0]['text'])
        self.gpt3_reasoning = response1['choices'][0]['text']
#         print('output tokens', len(self.gpt3_reasoning)/4)
        self.final_output['assertion'] = self.assertion
        self.final_output['retrieved_paras'] = self.ranked_paras
        self.final_output['reasoning'] = self.gpt3_reasoning
        return self.final_output


class process:
    
    def __init__(self):
        self.prepare_data_obj = PrepareData()
        
   
    def get_answer_extractiveQA(self, EQAreader,context, question):
        
        p = Pipeline()
        p.add_node(component=readerMem, name="Reader", inputs=["Query"])
        res = p.run(
            query=question, documents=[Document(content=context)],params={"top_k": 1})
        result = {"query" : res['query'],
                 "context" :res['documents'][0].content,
                 "answer":res['answers'][0].answer
                }
        return result, res
    
    def get_answer_generativeQA(self, generator,context, question):
        
        p = Pipeline()
        p.add_node(component=generator, name="Generator", inputs=["Query"])
        res = p.run(
            query=question, documents=[Document(content=context)],params={"top_k": 1})
        result = {"query" : res['query'],
                 "context" :res['documents'][0].content,
                 "answer":res['answers'][0].answer
                }
        return result, res
        
    def GPT3Query_1(self, protocol_number, query, retriever_param):
        
        self.prepare_data_obj = PrepareData()
        preprocessed_df = self.prepare_data_obj.protocolIngestion(protocol_number)
        main_df = self.prepare_data_obj.preprocessData(protocol_number, preprocessed_df)

        passage_retrieve_obj = Paragraph_Retrieval(main_df, query)
        final_df = passage_retrieve_obj.run_paragraph_retrieval()

        widget_new = widgets.HTML(value='<b> Please give me sometime while I find your answer using GPT.. </b>') 
        widget_new.add_class('lbl_bg')
        display(widget_new)        
        r = Reasoning(query, final_df, retriever_param)
        final_output = r.reason_w_GPT3()
        
        answer = final_output['reasoning']
        self.to_use_ip_df = pd.DataFrame()
        self.to_use_ip_df['Section_Name'] = ['*']*len(final_output['retrieved_paras'])
        self.to_use_ip_df['Content'] = ['*']*len(final_output['retrieved_paras'])
        
        for _ in range(len(final_output['retrieved_paras'])):
            self.to_use_ip_df['Section_Name'][_] = final_output['retrieved_paras'][_][0]
            self.to_use_ip_df['Content'][_] = final_output['retrieved_paras'][_][1]
        
        return answer, final_output['retrieved_paras'], self.to_use_ip_df
    
    def EQAhaystackQuery_1(self, query, retriever_param):
    
        context = ' '
        for i in list(self.to_use_ip_df['Content']):
            context = context + ' ' + i
            
        result, res = self.get_answer_extractiveQA(readerMem,context, query)
        widget_new = widgets.HTML(value='<b> Please give me sometime while I find your answer using ExtractiveQA.. </b>') 
        widget_new.add_class('lbl_bg')
        display(widget_new)
        
        return result, res 
    
    def haystackQuery_1(self, query, retriever_param):

        context = ' '
        for i in list(self.to_use_ip_df['Content']):
            context = context + ' ' + i
        result, res = self.get_answer_generativeQA(generator,context, query)    
        
        widget_new = widgets.HTML(value='<b> Please give me sometime while I find your answer using GenerativeQA.. </b>') 
        widget_new.add_class('lbl_bg')
        display(widget_new)
        
        return result, res
    

class UI:
    def __init__(self):
        self.process_obj = process()
    
    def display_ui_updated(self):

        style= {'description_width': 'initial'}
        items_layout = Layout(width='auto') 
        align_kw = dict(
            _css = (('.widget-label', 'min-width', '20ex'),),
            margin = '0px 0px 5px 10px'
        )
        
        widget1 = widgets.HTML(value='<b> ENTER QUERY PARAMETERS </b>')
        protocol_no = widgets.Dropdown(options=['000', '001'], value = '001', description='<b> Select Protocol Number: </b>',style = style, layout = items_layout)
        user_type = widgets.RadioButtons(options=['Patient', 'Site Personnel'], description='<b> Select user type: </b>',style = style, layout = items_layout)

        widget2 = widgets.HTML(value='<b> Number of matching content required from Protocol: </b>')

        slider = widgets.IntSlider(value=10,
                                      min=1,
                                      max=10,
                                      step=1,
                                      description='',
                                      disabled=False,
                                      continuous_update=False,
                                      orientation='horizontal',
                                      readout=True,
                                      readout_format='d',
                                  style = style)


        item1 = widgets.HBox([protocol_no], layout = Layout(width = '100%'))
        item1 = widgets.VBox([widget1, item1, user_type], layout = Layout(width = '100%'))

        item2 = widgets.HBox([widget2], layout = Layout(width = '100%'))
        item2 = widgets.VBox([widget2, slider], layout = Layout(width = '100%'))

        final_list = ['-Select-']
        # Create dropdown widget
        item_new = widgets.Dropdown(

            options = final_list,
            description='<b> Frequently Asked Question (FAQ): </b>', style = style, layout = items_layout, **align_kw, 
        )

        # Function to update final list
        def update_final_list(change):
            if change['new'] == 'Patient':
                item_new.options = Patient_query
            else:
                item_new.options = Site_query

        user_type.observe(update_final_list, names='value')

        item_new_ = widgets.Text(placeholder= 'Input query.....', description='<b> Please enter your query </b>', disabled=False, style = style, layout = items_layout, **align_kw)
        item_new.add_class('top_spacing')
        item_new.add_class('left_spacing')
        item_new_.add_class('top_spacing')

        item3 = widgets.HBox([item1], layout = Layout(width = '100%'))
        item4 = widgets.VBox([item_new_, item_new], layout = Layout(width = '100%'))
        item5 = widgets.VBox([item3, item4, item2], layout = Layout(width = '100%'))
        b1 = widgets.Button(description='Generate Response', button_style= 'success', layout= widgets.Layout(
                    width='35%', positioning = 'center'))#layout=Layout(positioning='right'))

        b1.layout.align_items = 'center' 
        b1.add_class("top_spacing")
        b1.add_class("button_style")
        h1 = widgets.Box([widgets.HTML(value= "<b style='color: white;font-family: Georgia; margin-top: 50%'> Prompt-based Answering system</b>").add_class('white_label')], layout=Layout(justify_content= 'space-around', width='100%', color='white'))
        
        img = widgets.Box([widgets.HTML(value= "<b style='font-size: 10px; font-family: Georgia; color: white; margin-left: 0%'>Powered by</b> <img src=https://i.kym-cdn.com/entries/icons/original/000/040/858/cover7.jpg style='width: 170px; height: 75px; margin-left: 0%; '>")], 
        layout=Layout(justify_content= 'space-around', width='20%', height='100%',margin_top= '0%'))
        heading_box = widgets.HBox([h1, img], layout = Layout(width = '100%'))
        heading_box.add_class("box_style1")
        
        
        heading_box.add_class("box_style1")
        display(heading_box)
        item6 = widgets.VBox([item5, b1], layout = Layout(width = '100%'))
        item6.add_class('box_style2')
        display(item6)
        out = widgets.Output()


        @out.capture()
        def processUIQuery(b):

            with out:
                clear_output()
                try:
                    
                    global Protocol_Number
                    global User_Type
                    global FAQ
                    global Query
                    global Params
                    global flag
                    
                    Protocol_Number = protocol_no.value
                    FAQ = item_new.value
                    Query = item_new_.value
                    Params = slider.value
                    
                    if FAQ!='-Select-':
                        GPT_starttime = time.time()
                        #process
                        GPTanswer, GPTmeta, df = self.process_obj.GPT3Query_1(Protocol_Number, FAQ, Params)
                        GPT_endtime = time.time()
                        GPT_delta = GPT_endtime - GPT_starttime
                        print('Time taken by GPT : ' + str(GPT_delta) + ' seconds')
                        EQ_starttime = time.time()
                        EQAanswer_, EQAdocuments_list = self.process_obj.EQAhaystackQuery_1(FAQ, Params)
                        EQendtime = time.time()
                        EQ_delta = EQendtime-EQ_starttime
                        print('Time taken by ExtractiveQA : ' + str(EQ_delta) + ' seconds')
                        GQ_starttime = time.time()
                        GQanswer, GQcontent = self.process_obj.haystackQuery_1(FAQ, Params)
                        GQ_endtime = time.time()
                        GQ_delta = GQ_endtime-GQ_starttime
                        print('Time taken by GenerativeQA : ' + str(GQ_delta) + ' seconds')
                    
                    else:
#                         GPTanswer, GPTmeta, df = self.process_obj.GPT3Query_1(Protocol_Number, Query, Params)
#                         EQAanswer_, EQAdocuments_list = self.process_obj.EQAhaystackQuery_1(Query, Params)
#                         GQanswer, GQcontent = self.process_obj.haystackQuery_1(Query, Params)
                        GPT_starttime = time.time()
                        #process
                        GPTanswer, GPTmeta, df = self.process_obj.GPT3Query_1(Protocol_Number, Query, Params)
                        GPT_endtime = time.time()
                        GPT_delta = GPT_endtime - GPT_starttime
                        print('Time taken by GPT : ' + str(GPT_delta) + ' seconds')
                        EQ_starttime = time.time()
                        EQAanswer_, EQAdocuments_list = self.process_obj.EQAhaystackQuery_1(Query, Params)
                        EQendtime = time.time()
                        EQ_delta = EQendtime-EQ_starttime
                        print('Time taken by ExtractiveQA : ' + str(EQ_delta) + ' seconds')
                        GQ_starttime = time.time()
                        GQanswer, GQcontent = self.process_obj.haystackQuery_1(Query, Params)
                        GQ_endtime = time.time()
                        GQ_delta = GQ_endtime-GQ_starttime
                        print('Time taken by GenerativeQA : ' + str(GQ_delta) + ' seconds')
                        
                    #GenerativeQA display
                    df = df.groupby('Section_Name')['Content'].agg('\n'.join).reset_index()
                    
                    HSanswer = GQanswer['answer']
                    if not HSanswer.endswith('.'):
                        split_answer = sent_tokenize(HSanswer)
                        HSanswer = ' '.join(split_answer[:-1])

                                        
                    HSanswer_display = widgets.HTML(value='<b> Answer to your question : </b>')
                    HSanswer_widget = widgets.HTML(value=HSanswer)
#                     HScontext = GQanswer['context']
#                     HScontext_display = widgets.HTML(value='<b> Answer context : </b>')
#                     HScontext_widget = widgets.HTML(value=HScontext)
                    HSreference_display = widgets.HTML(value='<b> Reference sections from Protocol : </b>')

                    item7 = widgets.VBox([HSanswer_display, HSanswer_widget, HSreference_display], layout = Layout(width = '100%'))
                    
                    HSContext = ' '
                    for i in range(len(df)):
                        HSContext = HSContext + '<b>Section Name : </b>' + df['Section_Name'][i] + '<br>' + '<b>Section Content : </b>' + df['Content'][i] + '<br>'

                    HSContext = HSContext.replace('\n', '<br>')

                    HSContext_display = widgets.HTML(value=HSContext)
                    
                    item8 = widgets.VBox([item7, HSContext_display], layout = Layout(width = '100%'))

                    #GPT3
                    
#                     if not GPTanswer.endswith('.'):
#                         split_answer = sent_tokenize(GPTanswer)
#                         GPTanswer = ' '.join(split_answer[:-1])

                    GPTanswer_display = widgets.HTML(value = '<b> Answer to your question : </b>')
                    GPTanswer_widget = widgets.HTML(value = GPTanswer)

                    
                    GPTreference_display = widgets.HTML(value='<b> Reference sections from Protocol : </b>')
                    item9 = widgets.VBox([GPTanswer_display, GPTanswer_widget, GPTreference_display], layout = Layout(width = '100%'))

                    GPTContext = ' '

                    for i in range(len(df)):
                       
                        GPTContext = GPTContext + '<b>Section Name : </b>' + df['Section_Name'][i] + '<br>' + '<b>Section Content : </b>' + df['Content'][i] + '<br>'

                    GPTcontext_display = widgets.HTML(value=GPTContext)       
                    item10 = widgets.VBox([item9, GPTcontext_display], layout = Layout(width = '100%'))

                    #EQA       
                    
                    EQanswer = EQAanswer_['answer']
                    EQAanswer_display = widgets.HTML(value='<b> Answer to your question : </b>')
                    EQAanswer_widget = widgets.HTML(value=EQanswer)
#                     EQcontext = EQAanswer_['context']
#                     EQcontext_display = widgets.HTML(value='<b> Answer context : </b>')
#                     EQcontext_widget = widgets.HTML(value=EQcontext)
                    EQAreference_display = widgets.HTML(value='<b> Reference sections from Protocol : </b>')

                    item11 = widgets.VBox([EQAanswer_display, EQAanswer_widget, EQAreference_display], layout = Layout(width = '100%'))
                    
                    EQAContext = ' '
                    for i in range(len(df)):
                        EQAContext = EQAContext + '<b>Section Name : </b>' + df['Section_Name'][i] + '<br>' + '<b>Section Content : </b>' + df['Content'][i] + '<br>'

                    EQAContext = EQAContext.replace('\n', '<br>')

                    EQAcontext_display = widgets.HTML(value=EQAContext)
                    
                    item12 = widgets.VBox([item11, EQAcontext_display], layout = Layout(width = '100%'))

                    tab_nest = widgets.Tab()
                    tab_nest.children = [item10, item8, item12]
                    tab_nest.titles = ('GPT3','GenerativeQA', 'ExtractiveQA')
                    tab_nest.set_title(0, 'GPT3')
                    tab_nest.set_title(1, 'GenerativeQA')
                    tab_nest.set_title(2, 'ExtractiveQA')
                    tab_nest.add_class("box_style3")
#                     tab_nest.add_class('box_style2')

                    display(tab_nest)

                except Exception as e:
                    print(e)   
                    traceback.print_exc()
                    widget_nafisa = widgets.HTML(value='<b> Please enter missing parameters! </b>')
                    widget_nafisa.add_class('lbl_bg')
                    display(widget_nafisa)

        b1.on_click(processUIQuery)

        display(out)


