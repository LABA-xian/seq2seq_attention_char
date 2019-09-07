# seq2seq_attention_char

利用seq2seq attention架構訓練

使用方式：

    test = seq2seq_Attention_all('P3_big_BN.h5', 'QA_all_char.txt') #參數說明第一個參數為儲存權重黨名稱，第二個參數為輸入txt檔
    test.train() #訓練模型

    model = test.create_model() #建立推理模型
    encoder_Inference, decoder_Inference = test.createAttentionInference(model)#建立推理模型
    while True:
        test_text = [input('【input Answer】 \n' )] #輸入問句
        result = x.run_model(test_text) #產生問句
        print('【output question】 \n', result) #列印問句

    while True:
        text = input('【input Answer】 \n' ) #輸入問句
        result = test.translate(text, encoder_Inference, decoder_Inference, True) #產生問句
        print('【output question】 \n')
        print(result) #列印問句
