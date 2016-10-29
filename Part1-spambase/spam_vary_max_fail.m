P_train = P_train_std; 
Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 
hiddenLayerSize = [10];
net = fitnet(hiddenLayerSize);
net.trainFcn = 'traingd'; 
net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 40/100;
net.divideParam.testRatio = 0/100;
net.trainParam.epochs = 500;
maxFail = 25;
maxFailArr = zeros(12,1);
count = 1;
missclassificationRate = zeros(12,1);
while maxFail <= 300
    net.trainParam.max_fail = maxFail;
    maxFailArr(count,1) = maxFail;
    [net tr] = train(net,P_train,T_train);
    [fields N] = size(T_test);
    neuralnetscore = net(Val.P);
    missclassificationRate(count,1) = sum(0.5*abs(T_test - neuralnetscore))/N;
    savePerformancePlot(['Performance_maxfail_',strrep(num2str(maxFail),'.','_')],tr);
    saveErrorHistogram(['ErrorHist_maxfail_',strrep(num2str(maxFail),'.','_')],gsubtract(T_train,net(P_train)));
    saveTrainStatePlot(['TrainState_maxfail_',strrep(num2str(maxFail),'.','_')],tr);
    saveRegressionPlot(['Reg_maxfail_',strrep(num2str(maxFail),'.','_')],T_train,net(P_train));
    maxFail = maxFail + 25;
    count = count + 1;
end

saveMissclassificationPlot('maxFail_missclassificationRate',maxFailArr,missclassificationRate);

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

function saveMissclassificationPlot(figureName,epochs,misclassificationRate)
    fileName = ['Figures\VaryAlpha\MissclassificationRate\',figureName];
    h = figure;
    plot(epochs,misclassificationRate);
    saveas(h,[fileName,'.jpg']);
end