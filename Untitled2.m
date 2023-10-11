net = googlenet;
inputSize = net.Layers(1).InputSize;

p = genpath('C:\Users\marks\桌面\data\data224');% 獲得資料夾data下所有子檔案的路徑，這些路徑存在字串p中，以';'分割
length_p = size(p,2);%字串p的長度
path = {};%建立一個單元陣列，陣列的每個單元中包含一個目錄
temp = [];
for i = 1:length_p %尋找分割符';'，一旦找到，則將路徑temp寫入path陣列中
    if p(i) ~= ';'
        temp = [temp p(i)];
    else 
        temp = [temp '\']; %在路徑的最後加入 '\'
        path = [path ; temp];
        temp = [];
    end
end  
clear p length_p temp;
%至此獲得data資料夾及其所有子資料夾（及子資料夾的子資料夾）的路徑，存於陣列path中。
%下面是逐一資料夾中讀取影象
file_num = size(path,1);% 子資料夾的個數
for i = 1:file_num
    file_path =  path{i}; % 影象資料夾路徑
    
    img_path_list = dir(strcat(file_path,'*.jpg'));
    img_num = length(img_path_list); %該資料夾中影象數量
    if img_num > 0
        for j = 1:img_num
            image_name = img_path_list(j).name;% 影象名
            image =  imread(strcat(file_path,image_name));
            %fprintf('%d %d %s\n',i,j,strcat(file_path,image_name));% 顯示正在處理的路徑和影象名


                image = imresize(image,[224,224]);
                imwrite(image,[file_path,image_name]);
           
        end
    end
end