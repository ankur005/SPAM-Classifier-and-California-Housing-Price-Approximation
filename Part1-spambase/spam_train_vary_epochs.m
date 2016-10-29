P_train = P_train_std; 
Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 
hiddenLayerSize = [10];
net = fitnet(hiddenLayerSize);
net.trainFcn = 'traingd'; 
net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;
net.divideParam.testRatio = 0/100;
net.trainParam.max_fail = 25;
epochs = transpose([10,100,200,300,400,500,600,700,800,900,1000]);
missclassificationRate = zeros(11,1);
for i = 1:11
    net.trainParam.epochs = epochs(i,1);
    [net tr] = train(net,P_train,T_train);
    [fields N] = size(T_test);
    neuralnetscore = sign(net(Val.P));
    missclassificationRate(i,1) = sum(0.5*abs(T_test - neuralnetscore))/N;
    savePerformancePlot(['Performance_epochs_',strrep(num2str(epochs(i)),'.','_')],tr);
    saveErrorHistogram(['ErrorHist_epochs_',strrep(num2str(epochs(i)),'.','_')],gsubtract(T_train,net(P_train)));
    saveTrainStatePlot(['TrainState_epochs_',strrep(num2str(epochs(i)),'.','_')],tr);
    saveRegressionPlot(['Reg_epochs_',strrep(num2str(epochs(i)),'.','_')],T_train,net(P_train));
end

saveMissclassificationPlot('lr_missclassificationRate',epochs,missclassificationRate);

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