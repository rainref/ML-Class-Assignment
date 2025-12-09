%type={'rebuttal_attention_test_202600','rebuttal_attention_test_110800','rebuttal_attention_test_132400'};%{'attention_model5_without_attention_W60_decay3_10000'};

%type={'rebuttal_V3-99500_test','rebuttal_V3-99500_validation','rebuttal_V3-105500_test','rebuttal_V3-105500_validation','rebuttal_V3-110500_test','rebuttal_V3-110500_validation'}
%type={'attention_model5_without_multi_scale_W60_decay3','attention_model5_without_attention_W60_decay3_45000' };
%type={'alex_115','alex_832','attention_115','attention_832','v3_115','v3_832'};
%'rebuttal_alex_test_181800','rebuttal_alex_validation_181800'
type={'no_atten'};%,'rebuttal_finetune_attention_rimone_5000'};
%root = '..\txt_result\12.20\';%final\rename\';
root = './after_process/';

for n =1:length(type)
    a=[];
    data_positive=[];
    data_negative=[];
    a = load([root,type{1,n},'.txt']);
    num_positive = 0;
    num_negative=0;
    for i =1:length(a)
        if a(i,2)==0
            num_positive= num_positive+1;
            data_positive(num_positive,3)=1-a(i,4);
            data_positive(num_positive,4)=a(i,4);
        end
        
        if a(i,2)==1
            num_negative= num_negative+1;
            data_negative(num_negative,3)=1-a(i,4);
            data_negative(num_negative,4)=a(i,4);
        end

    end

    save([root,type{1,n},'_0.mat'],'data_positive');
    save([root,type{1,n},'_1.mat'],'data_negative');
end