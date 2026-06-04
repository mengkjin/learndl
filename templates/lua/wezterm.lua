-- mkdir -p ~/.config/wezterm
-- touch ~/.config/wezterm/wezterm.lua
-- gedit ~/.config/wezterm/wezterm.lua

--- __      __      _
--- \ \    / /__ __| |_ ___ _ _ _ __
---  \ \/\/ / -_)_ /  _/ -_) '_| '  \
---   \_/\_/\___/__|\__\___|_| |_|_|_|
---
--- wezterm.lua - personal configuration

local wezterm = require("wezterm")
-- local mux = wezterm.mux
local act = wezterm.action
local config = {}

-- Use config builder object if possible
if wezterm.config_builder then
	config = wezterm.config_builder()
end

-- Set colorscheme
config.color_scheme = "Banana Blueberry"

-- Font settings
-- config.font = wezterm.font("0xProto Nerd Font")
config.font_size = 11

-- Window setting/ appearance
config.window_decorations = "TITLE|RESIZE"
config.window_background_opacity = 0.9
-- config.macos_window_background_blur = 10

config.window_padding = {
	left = "1cell",
	right = "1cell",
	top = "0.0cell",
	bottom = "0.5cell",
}

config.initial_rows = 40
config.initial_cols = 100

config.enable_scroll_bar = true
config.scrollback_lines = 5000
config.default_workspace = "main"

-- Dim inactive panes
--config.inactive_pane_hsb = {
--	saturation = 0.24,
--	brightness = 0.5,
--}

-- Tab bar
-- config.enable_tab_bar = false
config.use_fancy_tab_bar = true
config.tab_bar_at_bottom = false
config.status_update_interval = 1000
wezterm.on("update-status", function(window, pane)
	local basename = function(s)
		return string.gsub(s, "(.*[/\\])(.*)", "%2")
	end

	-- Current working directory
	local cwd = pane:get_current_working_dir()
	if cwd then
		if type(cwd) == "userdata" then
			cwd = basename(cwd.file_path)
		else
			cwd = basename(cwd)
		end
	else
		cwd = ""
	end

	-- Current command
	local cmd = pane:get_foreground_process_name()
	cmd = cmd and basename(cmd) or ""

	-- Time
	-- local time = wezterm.strftime("%H:%M")

	-- Right status
	window:set_right_status(wezterm.format({
		{ Text = wezterm.nerdfonts.md_folder .. "  " .. cwd },
		{ Text = " | " },
		{ Foreground = { Color = "#e0af68" } },
		{ Text = wezterm.nerdfonts.fa_code .. "  " .. cmd },
		"ResetAttributes",
		{ Text = " | " },
		{ Text = wezterm.nerdfonts.md_clock .. "  " .. time },
		{ Text = "  " },
	}))
end)

config.keys = {
  -- Windows / Linux
  -- Move cursor word-by-word (Left / Right)
  { key = 'LeftArrow', mods = 'CTRL', action = act.SendString '\x1bb' },
  { key = 'RightArrow', mods = 'CTRL', action = act.SendString '\x1bf' },
  -- Move to the START / END of the line (Maps Alt + LeftArrow to send Escape + b)
  { key = 'a', mods = 'ALT', action = act.SendString '\x1bb' },
  { key = 'e', mods = 'ALT', action = act.SendString '\x1bf' },
  -- Delete a WORD backwards (Maps Alt + Backspace to send Escape + Backspace)
  { key = 'w', mods = 'ALT', action = act.SendString '\x1b\x7f' },
  -- Clear screen and discard the scrollback buffer 
  { key = 'k', mods = 'CTRL|ALT', action = act.ClearScrollback 'ScrollbackAndViewport' },
  
  -- MacOS
  -- Move cursor word-by-word (Left / Right)
  { key = 'LeftArrow', mods = 'OPT', action = act.SendKey { key = 'b', mods = 'ALT' } },
  { key = 'RightArrow', mods = 'OPT', action = act.SendKey { key = 'f', mods = 'ALT' } },
  -- Clear screen and scrollback buffer using Command + K
  { key = 'k', mods = 'CMD', action = act.ClearScrollback 'ScrollbackAndViewport' },

}

-- Return the configuration to wezterm
return config
