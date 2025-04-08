#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import datetime

class Reporter:
    """實驗報告產生器。

    該類別提供了生成和更新實驗報告的功能。
    """

    def __init__(self, experiment_id=1, report_path="REPORT.md"):
        """初始化報告產生器。

        Args:
            experiment_id (int, optional): 實驗ID。預設為1。
            report_path (str, optional): 報告檔案路徑。預設為"REPORT.md"。
        """
        self.experiment_id = experiment_id
        self.report_path = report_path
        self.date = datetime.datetime.now().strftime("%Y%m%d")
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_experiment_data(self, data, output_path=None):
        """將實驗數據儲存為 JSON 檔案。

        Args:
            data (dict): 要儲存的實驗數據。
            output_path (str, optional): 輸出檔案路徑。如果為 None，則自動生成。

        Returns:
            str: 儲存的檔案路徑。
        """
        if output_path is None:
            # 取得函數名稱以用於檔名
            import inspect
            frame = inspect.currentframe().f_back
            func_name = frame.f_code.co_name
            output_path = f"ExperimentData_Exp{self.experiment_id}_{self.date}_{func_name}.json"
        
        # 添加元數據
        data['experiment_id'] = self.experiment_id
        data['date'] = self.date
        data['timestamp'] = self.current_time
        
        # 儲存數據
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path

    def update_report(self, title, content, image_paths=None, data_paths=None):
        """更新實驗報告檔案。

        Args:
            title (str): 本次實驗的標題。
            content (str): 實驗內容和結果的描述。
            image_paths (list, optional): 圖片檔案路徑列表。
            data_paths (list, optional): 數據檔案路徑列表。

        Returns:
            bool: 更新成功返回 True，失敗返回 False。
        """
        try:
            # 檢查報告檔案是否存在，如果不存在則創建一個
            if not os.path.exists(self.report_path):
                with open(self.report_path, 'w', encoding='utf-8') as f:
                    f.write("# 音訊 STFT/iSTFT 處理實驗報告\n\n")
            
            # 讀取現有報告
            with open(self.report_path, 'r', encoding='utf-8') as f:
                existing_report = f.read()
            
            # 產生新的實驗報告條目
            new_entry = f"\n## 實驗 #{self.experiment_id} - {title} - {self.current_time}\n\n"
            new_entry += f"{content}\n\n"
            
            # 添加圖片
            if image_paths:
                new_entry += "### 圖形結果\n\n"
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    new_entry += f"![{img_name}](./{img_path})\n\n"
            
            # 添加數據檔案連結
            if data_paths:
                new_entry += "### 相關檔案\n\n"
                for data_path in data_paths:
                    data_name = os.path.basename(data_path)
                    new_entry += f"- [{data_name}](./{data_path})\n"
                new_entry += "\n"
            
            # 添加分隔線
            new_entry += "---\n"
            
            # 寫入更新後的報告
            with open(self.report_path, 'w', encoding='utf-8') as f:
                f.write(existing_report + new_entry)
            
            return True
        except Exception as e:
            print(f"Error updating report: {str(e)}")
            return False
