function [] = m_showboxes(im,pick_rois)

pick_rois = double(pick_rois);  % text函数要求double类型
L = size(pick_rois,1);
color = {[1,0,0],[1,0.281250000000000,0],[1,0.562500000000000,0],[1,0.843750000000000,0],[0.875000000000000,1,0],[0.593750000000000,1,0],[0.312500000000000,1,0],[0.0312500000000000,1,0],[0,1,0.250000000000000],[0,1,0.531250000000000]};
imshow(im);
for i = 1:L
    rectangle('position',[pick_rois(i,1),pick_rois(i,2),pick_rois(i,3)-pick_rois(i,1),pick_rois(i,4)-pick_rois(i,2)], ...
            'LineWidth',4,'EdgeColor',color{pick_rois(i,6)});
%     switch pick_rois(i,6)
%         case 1  
%             classname = '2S1';
%         case 2
%             classname = 'BMP2';
%         case 3
%             classname = 'BRDM2';
%         case 4 
%             classname = 'BTR60';
%         case 5
%             classname = 'BTR70';
%         case 6 
%             classname = 'D7';
%         case 7 
%             classname = 'T62';
%         case 8 
%             classname = 'T72';
%         case 9 
%             classname = 'ZIL131';
%         case 10
%             classname = 'ZSU234';
%     end
%     recog_rate = uint8(pick_rois(i,5)*100);
    % 显示类别及预测概率
    text(pick_rois(i,1),pick_rois(i,2)-15,[num2str(pick_rois(i,6)) ':' num2str(pick_rois(i,5),'%.3f')],'FontSize',13,'BackgroundColor', 'w');
end