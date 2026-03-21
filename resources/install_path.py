#!/usr/bin/env python3
# coding: utf-8
"""
Multi-platform Python path installer
Supports Windows, macOS, Ubuntu, etc.
Detect shell type and set PYTHONPATH environment variable
"""

import os
import sys
import platform
import socket
import subprocess
from pathlib import Path

class PathInstaller:
    PATH_DICT = {
        'mengkjin-server':  '/home/mengkjin/workspace/learndl',
        'HST-jinmeng':      'E:/workspace/learndl',
        'Mathews-Mac':      '/Users/mengkjin/workspace/learndl',
    }
    def __init__(self):
        self.project_name = 'learndl'
        self.machine_name = self._get_machine_name()
        self.system = platform.system().lower()
        self.project_root = self._get_project_root()
        self.shell = self._detect_shell()

    def _get_machine_name(self):
        """get machine name"""
        return socket.gethostname().split('.')[0]

    def _get_project_root(self):
        """get project root"""
        return str(Path(self.PATH_DICT[self._get_machine_name()]).absolute())
    
    def _detect_shell(self):
        """detect current used shell"""
        if self.system == 'windows':
            return 'powershell' if 'powershell' in os.environ.get('SHELL', '').lower() else 'cmd'
        else:
            shell = os.environ.get('SHELL', '').lower()
            if 'zsh' in shell:
                return 'zsh'
            elif 'bash' in shell:
                return 'bash'
            elif 'fish' in shell:
                return 'fish'
            else:
                return 'bash'  # default use bash
    
    def _get_config_files(self):
        """get config files to modify"""
        home = Path.home()
        config_files = []
        
        if self.system == 'windows':
            # Windows environment variable
            config_files.append({
                'type': 'windows_env',
                'path': 'HKEY_CURRENT_USER\\Environment',
                'description': 'Windows user environment variable'
            })
        else:
            # Unix-like system config file
            shell_configs = {
                'bash': ['.bashrc', '.bash_profile', '.profile'],
                'zsh': ['.zshrc', '.zprofile'],
                'fish': ['.config/fish/config.fish']
            }
            
            for config_file in shell_configs.get(self.shell, ['.bashrc']):
                config_path = home / config_file
                if config_path.exists() or config_file in ['.bashrc', '.zshrc']:
                    config_files.append({
                        'type': 'shell_config',
                        'path': str(config_path),
                        'description': f'{self.shell}配置文件'
                    })
        
        return config_files
    
    def _add_to_shell_config(self, config_file):
        """add to shell config file"""
        config_path = Path(config_file['path'])
        
        # prepare to add content
        pythonpath_line = f'export PYTHONPATH="{self.project_root}:$PYTHONPATH"'
        
        # check if already exists
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if self.project_root in content:
                print(f"⚠️  {config_file['description']} already has project path")
                return True
        
        # create directory (if not exists)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # add path setting
        with open(config_path, 'a', encoding='utf-8') as f:
            f.write(f'\n# Python path setting - added by install_path.py\n')
            f.write(f'# project: {self.project_name}\n')
            f.write(f'{pythonpath_line}\n')
        
        print(f"✓ added to {config_file['description']}: {config_path}")
        return True
    
    def _add_to_windows_env(self):
        """add to Windows environment variable"""
        try:
            import winreg
            
            # open environment variable registry
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS)
            
            try:
                # get existing PYTHONPATH
                current_path, _ = winreg.QueryValueEx(key, "PYTHONPATH")
                if self.project_root not in current_path:
                    new_path = f"{self.project_root};{current_path}" if current_path else self.project_root
                    winreg.SetValueEx(key, "PYTHONPATH", 0, winreg.REG_EXPAND_SZ, new_path)
                    print(f"✓ added to Windows environment variable: {self.project_root}")
                else:
                    print("⚠️  Windows environment variable already has project path")
            except FileNotFoundError:
                # PYTHONPATH does not exist, create new
                winreg.SetValueEx(key, "PYTHONPATH", 0, winreg.REG_EXPAND_SZ, self.project_root)
                print(f"✓ created Windows environment variable: {self.project_root}")
            finally:
                winreg.CloseKey(key)
            
            return True
        except ImportError:
            print("❌ cannot modify Windows environment variable, please set manually")
            return False
        except Exception as e:
            print(f"❌ failed to modify Windows environment variable: {e}")
            return False
    
    def install(self):
        """install path setting"""
        print(f"🚀 start installing Python path setting...")
        print(f"📁 project root: {self.project_root}")
        print(f"💻 operating system: {self.system}")
        print(f"🐚 shell type: {self.shell}")
        print("-" * 50)
        
        success_count = 0
        config_files = self._get_config_files()
        
        for config_file in config_files:
            try:
                if config_file['type'] == 'windows_env':
                    if self._add_to_windows_env():
                        success_count += 1
                elif config_file['type'] == 'shell_config':
                    if self._add_to_shell_config(config_file):
                        success_count += 1
            except Exception as e:
                print(f"❌ failed to configure {config_file['description']}: {e}")
        
        print("-" * 50)
        if success_count > 0:
            print(f"✅ installation completed! successfully configured {success_count} files")
            print("🔄 please restart terminal or run 'source ~/.bashrc' (bash) or 'source ~/.zshrc' (zsh)")
            self._verify_installation()
        else:
            print("❌ installation failed, please check permissions or set manually")
    
    def _verify_installation(self):
        """verify installation result"""
        print("\n🔍 verify installation result...")
        
        # check PYTHONPATH environment variable
        pythonpath = os.environ.get('PYTHONPATH', '')
        if self.project_root in pythonpath:
            print("✅ PYTHONPATH environment variable is set")
        else:
            print("⚠️  PYTHONPATH environment variable is not valid , please restart terminal")
    
    def uninstall(self):
        """uninstall path setting"""
        print(f"🗑️  start uninstalling Python path setting...")
        print(f"📁 project root: {self.project_root}")
        print("-" * 50)
        
        config_files = self._get_config_files()
        success_count = 0
        
        for config_file in config_files:
            try:
                if config_file['type'] == 'shell_config':
                    if self._remove_from_shell_config(config_file):
                        success_count += 1
                elif config_file['type'] == 'windows_env':
                    if self._remove_from_windows_env():
                        success_count += 1
            except Exception as e:
                print(f"❌ failed to uninstall from {config_file['description']}: {e}")
        
        print("-" * 50)
        if success_count > 0:
            print(f"✅ uninstall completed! successfully removed configuration from {success_count} files")
        else:
            print("⚠️  no configuration found to uninstall")
    
    def _remove_from_shell_config(self, config_file):
        """remove from shell config file"""
        config_path = Path(config_file['path'])
        
        if not config_path.exists():
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # filter out lines containing project path
        new_lines = []
        skip_next = False
        
        for line in lines:
            if skip_next:
                skip_next = False
                continue
            
            if self.project_root in line and ('PYTHONPATH' in line or 'Python path setting' in line):
                skip_next = True  # skip next line (usually a comment)
                continue
            
            new_lines.append(line)
        
        # write back to file
        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"✓ 已从 {config_file['description']} 移除: {config_path}")
        return True
    
    def _remove_from_windows_env(self):
        """remove from Windows environment variable"""
        try:
            import winreg
            
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS)
            
            try:
                current_path, _ = winreg.QueryValueEx(key, "PYTHONPATH")
                if self.project_root in current_path:
                    # remove project path
                    paths = current_path.split(';')
                    paths = [p for p in paths if p != self.project_root]
                    new_path = ';'.join(paths) if paths else ''
                    
                    if new_path:
                        winreg.SetValueEx(key, "PYTHONPATH", 0, winreg.REG_EXPAND_SZ, new_path)
                    else:
                        winreg.DeleteValue(key, "PYTHONPATH")
                    
                    print("✓ removed from Windows environment variable")
                    return True
                else:
                    print("⚠️  Windows environment variable does not have project path")
                    return False
            except FileNotFoundError:
                print("⚠️  PYTHONPATH environment variable does not exist")
                return False
            finally:
                winreg.CloseKey(key)
        except Exception as e:
            print(f"❌ failed to remove from Windows environment variable: {e}")
            return False

def main():
    installer = PathInstaller()
    
    print(sys.argv)
    if len(sys.argv) > 1 and sys.argv[1] == 'uninstall':
        installer.uninstall()
    else:
        installer.install()

    try:
        from src.proj import MACHINE
        python_script = os.path.join(installer.project_root, 'src' , 'scripts' , '0_check' , '0_test_streamlit.py')
        python_root = MACHINE.python_path
        subprocess.run([python_root, python_script])
    except Exception as e:
        print(f'error: {e}')

if __name__ == '__main__':
    main()
