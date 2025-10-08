import dropbox
import os
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", default="download", choices=["download", "upload"], help="download or upload")
    parser.add_argument("--token_file", default=".env", help="The file path of the dropbox access token")
    parser.add_argument("--local_dir", default="models/self_training", help="The directory path of the folder in local place")
    parser.add_argument("--dropbox_dir", default="/paper/self_training", help="The directory path of the folder in dropbox place")
    return parser.parse_args()

if __name__ == "__main__":
    # 你的 Dropbox Access Token
    args = parser()
    with open(args.token_file, 'r') as f:
        ACCESS_TOKEN = f.read()
    
    # 建立 Dropbox 物件
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    
    if args.action == "upload":
        # 遞迴走訪資料夾
        for root, dirs, files in os.walk(args.local_dir):
            for filename in files:
                # 本機完整檔案路徑
                local_file_path = os.path.join(root, filename)

                # 計算相對路徑 → 對應到 Dropbox 上的路徑
                relative_path = os.path.relpath(local_file_path, args.local_dir)
                dropbox_path = os.path.join(args.dropbox_dir, relative_path).replace("\\", "/")

                # 上傳
                with open(local_file_path, "rb") as f:
                    print(f"⬆️ 上傳中: {dropbox_path}")
                    dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))

        print("✅ 資料夾上傳完成！")
    else:
        def download_folder(dbx, dropbox_path, local_path):
            """
            遞迴下載 Dropbox 資料夾內容到本機
            """
            try:
                # 取得資料夾內容
                result = dbx.files_list_folder(dropbox_path)

                while True:
                    for entry in result.entries:
                        # 如果是資料夾就遞迴
                        if isinstance(entry, dropbox.files.FolderMetadata):
                            # 建立本機資料夾
                            new_local_folder = os.path.join(local_path, entry.name)
                            os.makedirs(new_local_folder, exist_ok=True)
                            # 遞迴呼叫下載該資料夾
                            download_folder(dbx, entry.path_lower, new_local_folder)

                        # 如果是檔案就下載
                        elif isinstance(entry, dropbox.files.FileMetadata):
                            local_file_path = os.path.join(local_path, entry.name)
                            print(f"⬇️ 下載中: {entry.path_lower} 到 {local_file_path}")
                            metadata, res = dbx.files_download(entry.path_lower)
                            with open(local_file_path, 'wb') as f:
                                f.write(res.content)

                    if not result.has_more:
                        break
                    result = dbx.files_list_folder_continue(result.cursor)

            except dropbox.exceptions.ApiError as e:
                print(f"API 錯誤: {e}")
        os.makedirs(args.local_dir, exist_ok=True)
        download_folder(dbx, args.dropbox_dir, args.local_dir)
        print("✅ 資料夾下載完成！")
