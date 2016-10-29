P_train = P_train_std; 
Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 
%hiddenLayerSize = [1,5,10,15,20,25,30,35,40,45,50,55,57,65,70,75,80,85,90,95,100];
hiddenLayerSize = [40,80];
missclassificationRate = zeros(2,1);
for i = 1:2
    net = fitnet(hiddenLayerSize(i));
    net.trainFcn = 'trainbr'; 
    net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 30/100;
    net.divideParam.testRatio = 0/100;
    net.trainParam.epochs =200;
    net.trainParam.max_fail = 25;
    [net tr] = train(net,P_train,T_train);
    [fields N] = size(T_test);
    neuralnetscore = sign(net(Val.P));
    missclassificationRate = sum(0.5*abs(T_test - neuralnetscore))/N;
    disp(missclassificationRate);
    savePerformancePlot(['Performance_br_',num2str(hiddenLayerSize(i))],tr);
    saveErrorHistogram(['ErrorHist_br_',num2str(hiddenLayerSize(i))],gsubtract(T_train,net(P_train)));
    saveTrainStatePlot(['TrainState_br_',num2str(hiddenLayerSize(i))],tr);
    saveRegressionPlot(['Reg_br_',num2str(hiddenLayerSize(i))],T_train,net(P_train));
end

% saveMissclassificationPlot('hidlay_size_missclassificationRate_lm',hiddenLayerSize,missclassificationRate);

function savePerformancePlot(figureName,tr)
    fileName = ['Figures\Performance\',figureName];
    h = figure;
    plotperform(tr);
    saveas(h,[fileName,'.jpg']);
end

function saveErrorHistogram(figureName,graphInput)
    fileName = ['Figures\ErrorHistogram\',figureName];
    h = figure;
    ploterrhist(graphInput);
    saveas(h,[fileName,'.jpg']);
end

function saveRegressionPlot(figureName,T_train, output)
    fileName = ['Figures\Regression\',figureName];
    h = figure;
    plotregression(T_train,output);
    saveas(h,[fileName,'.jpg']);
end

function saveTrainStatePlot(figureName,tr)
    fileName = ['Figures\TrainState\',figureName];
    h = figure;
    plottrainstate(tr);
    saveas(h,[fileName,'.jpg']);
end