P_train = P_train_std; 
T_train = T_train_std; Val = Val_std;
hiddenLayerSize = [20];
learningRate = transpose([0.001,0.003,0.01,0.03,0.1,0.3,0.9]);
RMS_Error = zeros(7,1);
for i = 1:size(learningRate)
    net = fitnet(hiddenLayerSize);
    net.trainFcn = 'traingd'; 
    net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 30/100;
    net.divideParam.testRatio = 0/100;
    net.trainParam.epochs = 200;
    net.trainParam.max_Fail = 25;
    net.trainParam.lr = learningRate(i);
    [net tr] = train(net,P_train,T_train);
    [fields N] = size(T_test);
    
    est = net(Val.P);
    est = mapstd('apply', est, TS_train_std); %%% Use this line if you use STD or PCA preprocessing on the data. IMPORTANT: Uncomment the corresponding line above 
    RMS_Error(i) = perform(net, T_test, est); % equivalent to sqrt(mean((T_test - est).^2));
    savePerformancePlot(['Performance_lr',strrep(num2str(learningRate(i)),'.','_')],tr);
    saveErrorHistogram(['ErrorHist_lr',strrep(num2str(learningRate(i)),'.','_')],gsubtract(T_train,net(P_train)));
    saveTrainStatePlot(['TrainState_lr',strrep(num2str(learningRate(i)),'.','_')],tr);
    saveRegressionPlot(['Reg__lr',strrep(num2str(learningRate(i)),'.','_')],T_train,net(P_train));
end
saveMissclassificationPlot('hiddenlayersize__lr',learningRate,RMS_Error);

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

function saveMissclassificationPlot(figureName,epochs,misclassificationRate)
    fileName = ['Figures\RMSError\',figureName];
    h = figure;
    plot(epochs,misclassificationRate);
    saveas(h,[fileName,'.jpg']);
end