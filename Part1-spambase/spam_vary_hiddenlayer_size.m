P_train = P_train_std; 
Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 
%hiddenLayerSize = [1,5,10,15,20,25,30,35,40,45,50,55,57,65,70,75,80,85,90,95,100];
hiddenLayerSize = [10,20,30,40,50,57,70,80,90,100];
missclassificationRate = zeros(10,1);
for i = 1:10
    net = fitnet(hiddenLayerSize(i));
    net.trainFcn = 'trainlm'; 
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
    missclassificationRate(i) = sum(0.5*abs(T_test - neuralnetscore))/N;
    savePerformancePlot(['Performance_lm',num2str(hiddenLayerSize(i))],tr);
    saveErrorHistogram(['ErrorHist_lm',num2str(hiddenLayerSize(i))],gsubtract(T_train,net(P_train)));
    saveTrainStatePlot(['TrainState_lm',num2str(hiddenLayerSize(i))],tr);
    saveRegressionPlot(['Reg_lm',num2str(hiddenLayerSize(i))],T_train,net(P_train));
end

saveMissclassificationPlot('hidlay_size_missclassificationRate_lm',hiddenLayerSize,missclassificationRate);

function savePerformancePlot(figureName,tr)
    fileName = ['Figures\VaryAlpha\Performance\',figureName];
    h = figure;
    plotperform(tr);
    saveas(h,[fileName,'.jpg']);
end

function saveErrorHistogram(figureName,graphInput)
    fileName = ['Figures\VaryAlpha\ErrorHistogram\',figureName];
    h = figure;
    ploterrhist(graphInput);
    saveas(h,[fileName,'.jpg']);
end

function saveRegressionPlot(figureName,T_train, output)
    fileName = ['Figures\VaryAlpha\Regression\',figureName];
    h = figure;
    plotregression(T_train,output);
    saveas(h,[fileName,'.jpg']);
end

function saveTrainStatePlot(figureName,tr)
    fileName = ['Figures\VaryAlpha\TrainState\',figureName];
    h = figure;
    plottrainstate(tr);
    saveas(h,[fileName,'.jpg']);
end

function saveMissclassificationPlot(figureName,hiddenLayerSize,misclassificationRate)
    fileName = ['Figures\VaryAlpha\MissclassificationRate\',figureName];
    h = figure;
    plot(hiddenLayerSize,misclassificationRate);
    saveas(h,[fileName,'.jpg']);
end