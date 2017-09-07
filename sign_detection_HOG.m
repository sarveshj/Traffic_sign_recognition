
%{
clc;
clear all;
close all;


%step 1: read specific folders -- specific target data
%step2: each image, resize it to 64x64 and train svm
%step3: perform classification


data_path='/sarvesh/umcp/enpm_673_robot_perception/projects/project2/part2/Training-20170329T192257Z-001/'

train_data_path=fullfile(data_path,'Training');
test_data_path=fullfile(data_path,'Testing');

class_name={'00045','00021','00038','00035','00017','00001','00014','00019'};

training_features=[];
training_labels=[];

testing_features=[];
testing_labels=[];
jj=0;


%training
for index=1:size(class_name,2)

    %which class are you processing
    sprintf('process class -- %d',index)
    
    %process each class data
     class_data=fullfile(train_data_path,class_name{1,index});
     image_files_list=dir([class_data '/*.ppm']);
    
     
    
     training_labels= [training_labels;  index.*ones(size(image_files_list,1),1)];
     
     % process each image of a particular class
     for ii=1:length(image_files_list)
        jj=jj+1;
        %file_name=image_files_list(ii).name;
        image_path=fullfile(train_data_path,class_name,image_files_list(ii).name);
        image=imread(image_path{1,index}); 
        
               
        
        %-------------------------------------------------------------
        %step 2: resize, extract HOG and train SVM
        %-------------------------------------------------------------
        
        
        resize_image=imresize(image,[64 64]);
        resize_image=imbinarize(rgb2gray(resize_image));
        
        %[hog_features,viz]= extractHOGFeatures(resize_image,'CellSize',[5 5]);
        
        training_features(jj,:)= extractHOGFeatures(resize_image,'CellSize',[5 5]);
       
        
     end     
    
end
classifier=fitcecoc(training_features,training_labels);

%}
%error('completed training ....')


%-----------
% testing
%-----------

close all
test_video_frame='/media/sarveshj/UBUNTU 16_0/Output_new/out_part2_orig';
out_path='/sarvesh/Dropbox/codeRespositoryGit/gitRepository/enpm_673_robot_perception/project_2/out_part2_sign_recog';
image_files_list=dir([test_video_frame '/*.jpg']);
   
%for ii=1000:1200%size(image_files_list)
for ii=1:size(image_files_list)
%for ii=1
       %initialize label
        predictedLabels={};
        %keep track of total number of training images
        
        %file_name='image.034067.jpg';
        
        file_name=image_files_list(ii).name;
        sprintf('predicting for %s',image_files_list(ii).name)
        % force the ordering
        
        %image_path=fullfile(test_video_frame,strcat(int2str(ii),'.jpg'));
        image_path=fullfile(test_video_frame,file_name);
        image=imread(image_path); 
        %imshow(image);
        %error('done')
        
        
        %-------------------------------------------------------------
        %step 2: resize, extract HOG and train SVM
        %-------------------------------------------------------------
        
        
        %resize_image=imresize(image,[64 64]);
        %resize_image=imbinarize(rgb2gray(resize_image));
        
         
        % get the bounding boxes
        bbox_list =function_compute_bbox(image);
        
               
        if isempty(bbox_list)
            %error('do you want to save when nothing is detected???')
            disp('no bounding box detected ...');
            savePath=fullfile(out_path,file_name);
            imwrite(image, savePath );
        else
            
        
                %for each bounding box train, as a class
                for bbox_index=1:size(bbox_list,1)
                     %bb_image=imcrop()
                     %error('found....')

                     y_start_lim=bbox_list(bbox_index,2);
                    y_end_lim=bbox_list(bbox_index,4);
                    x_start_lim=bbox_list(bbox_index,1);
                    x_end_lim=bbox_list(bbox_index,3);

                     %bb_image=image(floor(bbox_list(bbox_index,2)):floor(bbox_list(bbox_index,4)+1), ...
                        %  floor(bbox_list(bbox_index,1)): floor(bbox_list(bbox_index,3))+1);

                      bb_image=image(floor(y_start_lim:y_start_lim+y_end_lim+1) ,floor(x_start_lim:x_start_lim+x_end_lim+1));
                      %imshow(bb_image)
                      %error('dddddd')
                      
                      %rescale the bounding box for HOG feature extraction
                      bb_image=imresize(bb_image,[64 64]);
                      
                      %compute hog feature and add to list

                    testing_features= extractHOGFeatures(bb_image,'CellSize',[5 5]);

                    label=class_name{1,predict(classifier, testing_features)};   
                    predictedLabels{bbox_index} =  label;
                
                end
                    
                
                    f=figure;
                    set(f,'Visible','off');
                    
                    imshow(insertObjectAnnotation(image, 'rectangle',bbox_list,predictedLabels ,...
                                                                 'FontSize',22, ...
                                                                'TextBoxOpacity',0.8));
                                                               
                     %title('Show Detected Sign')
                       
            

                savePath=fullfile(out_path,file_name);
                saveas(f, savePath );
            %error('doneeeeeeeeeeeeeeee')
        
        
        end
        
       %catch
           
           %continue
        
       %end
        
       
     end    
    


%train it !!!!
%}


%-----------------------------------------------------------
%testing
%----------------------------------------------------------






%dictedLabels = predict(classifier, testing_features);
%confMat = confusionmat(testing_labels, predictedLabels);
%helperDisplayConfusionMatrix(confMat)