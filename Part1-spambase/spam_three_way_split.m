P_train = P_train_std; 
Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 

hiddenLayerSize = [10];
net = fitnet(hiddenLayerSize);
net.trainFcn = 'traingd'; 
net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;
net.trainParam.epochs =500;
net.trainParam.max_fail = 25;
[net tr] = train(net,P_train,T_train);
[fields N] = size(T_test);
neuralnetscore = sign(net(Val.P));
missclassificationRate = sum(0.5*abs(T_test - neuralnetscore))/N;
disp(missclassificationRate);
savePerformancePlot(['Performance_threeway_80-10-10'],tr);
saveErrorHistogram(['ErrorHist_threeway_80-10-10'],gsubtract(T_train,net(P_train)));
saveTrainStatePlot(['TrainState_threeway_80-10-10'],tr);
saveRegressionPlot(['Reg_threeway_80-10-10'],T_train,net(P_train));

%saveMissclassificationPlot('lr_missclassificationRate',lr,missclassificationRate);

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