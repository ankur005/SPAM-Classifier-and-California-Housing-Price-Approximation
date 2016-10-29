P_train = P_train_std; 
T_train = T_train_std; Val = Val_std;
hiddenLayerSize = [10];
maxFail = transpose([25,100,200]);
RMS_Error = zeros(3,1);
for i = 1:size(maxFail)
    net = fitnet(hiddenLayerSize);
    net.trainFcn = 'trainlm'; 
    net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 30/100;
    net.divideParam.testRatio = 0/100;
    net.trainParam.epochs = 400;
    net.trainParam.max_Fail = maxFail(i);
    [net tr] = train(net,P_train,T_train);
    [fields N] = size(T_test);
    
    est = net(Val.P);
    est = mapstd('apply', est, TS_train_std); %%% Use this line if you use STD or PCA preprocessing on the data. IMPORTANT: Uncomment the corresponding line above 
    RMS_Error = perform(net, T_test, est); % equivalent to sqrt(mean((T_test - est).^2));
    disp(num2str((RMS_Error),'%.10f'));
    savePerformancePlot(['Performance_max_fail',strrep(num2str(maxFail(i)),'.','_')],tr);
    saveErrorHistogram(['ErrorHist_max_fail',strrep(num2str(maxFail(i)),'.','_')],gsubtract(T_train,net(P_train)));
    saveTrainStatePlot(['TrainState_max_fail',strrep(num2str(maxFail(i)),'.','_')],tr);
    saveRegressionPlot(['Reg__max_fail',strrep(num2str(maxFail(i)),'.','_')],T_train,net(P_train));
end
%saveMissclassificationPlot('hiddenlayersize__6040_RMSError',hiddenLayerSize,RMS_Error);

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