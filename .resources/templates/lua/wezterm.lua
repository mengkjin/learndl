-- config path:
-- ~/.config/wezterm/wezterm.lua in linux / macos
-- %USERPROFILE%\.config\wezterm\wezterm.lua in windows

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

local function wezterm_binary()
  if wezterm.target_triple:find("windows") then
    return wezterm.executable_dir .. "\\wezterm.exe"
  end
  return wezterm.executable_dir .. "/wezterm"
end

local function build_merge_tab_choices(mux_window)
  local choices = {}
  for _, info in ipairs(mux_window:tabs_with_info()) do
    if not info.is_active then
      local tab = info.tab
      local active_pane = tab:active_pane()
      local title = active_pane:get_title() or tab:get_title() or "untitled"
      table.insert(choices, {
        id = tostring(active_pane:pane_id()),
        label = string.format("Tab %d: %s", info.index + 1, title),
      })
    end
  end
  return choices
end

-- Use config builder object if possible
if wezterm.config_builder then
	config = wezterm.config_builder()
end

-- Use local unix domain multiplexer by default
config.default_mux_server_domain = 'local'
config.leader = { key = 'a', mods = 'CTRL', timeout_milliseconds = 1000 }

-- Set colorscheme
config.color_scheme = "Banana Blueberry"

-- Font settings
-- config.font = wezterm.font("0xProto Nerd Font")
config.font_size = $FONT_SIZE$

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
config.status_update_interval = 200
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
	local time = wezterm.strftime("%H:%M")

	local status = {}

	if window:leader_is_active() then
		table.insert(status, { Attribute = { Intensity = "Bold" } })
		table.insert(status, { Background = { Color = "#e0af68" } })
		table.insert(status, { Foreground = { Color = "#1a1b26" } })
		table.insert(status, { Text = " ⚡ LEADER WAITING " })
		table.insert(status, "ResetAttributes")
		table.insert(status, { Text = " | " })
	end

	table.insert(status, { Text = wezterm.nerdfonts.md_folder .. "  " .. cwd })
	table.insert(status, { Text = " | " })
	table.insert(status, { Foreground = { Color = "#e0af68" } })
	table.insert(status, { Text = wezterm.nerdfonts.fa_code .. "  " .. cmd })
	table.insert(status, "ResetAttributes")
	table.insert(status, { Text = " | " })
	table.insert(status, { Text = wezterm.nerdfonts.md_clock .. "  " .. time })
	table.insert(status, { Text = "  " })

	window:set_right_status(wezterm.format(status))
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
  { key = 'k', mods = 'CTRL', action = act.ClearScrollback 'ScrollbackAndViewport' },
  
  -- MacOS
  -- Move cursor word-by-word (Left / Right)
  { key = 'LeftArrow', mods = 'OPT', action = act.SendKey { key = 'b', mods = 'ALT' } },
  { key = 'RightArrow', mods = 'OPT', action = act.SendKey { key = 'f', mods = 'ALT' } },
  -- Clear screen and scrollback buffer using Command + K
  { key = 'k', mods = 'CMD', action = act.ClearScrollback 'ScrollbackAndViewport' },

  -- Spawn a new tab
  {
    key = 't',
    mods = 'LEADER',
    action = wezterm.action.SpawnTab 'CurrentPaneDomain',
  },
  -- Split pane horizontally
  {
    key = '"',
    mods = 'LEADER|SHIFT',
    action = wezterm.action.SplitHorizontal { domain = 'CurrentPaneDomain' },
  },
  -- Split pane vertically
  {
    key = '%',
    mods = 'LEADER|SHIFT',
    action = wezterm.action.SplitVertical { domain = 'CurrentPaneDomain' },
  },
  -- Navigate between panes
  {
    key = 'j',
    mods = 'LEADER',
    action = wezterm.action.ActivatePaneDirection 'Left',
  },
  {
    key = 'l',
    mods = 'LEADER',
    action = wezterm.action.ActivatePaneDirection 'Right',
  },
  {
    key = 'i',
    mods = 'LEADER',
    action = wezterm.action.ActivatePaneDirection 'Up',
  },
  {
    key = 'k',
    mods = 'LEADER',
    action = wezterm.action.ActivatePaneDirection 'Down',
  },
  -- Press LEADER then 'v' or 'h' to pull another tab's pane into the current tab as a split vertically or horizontally
  {
    key = 'v',
    mods = 'LEADER',
    action = wezterm.action_callback(function(window, pane)
      local mux_window = window:mux_window()
      if not mux_window then
        return
      end

      local choices = build_merge_tab_choices(mux_window)
      if #choices == 0 then
        window:toast_notification('merge tab', 'No other tabs to merge', nil, 3000)
        return
      end

      window:perform_action(
        act.InputSelector {
          title = "Select Tab to Convert into a Split Pane",
          fuzzy = true,
          choices = choices,
          action = wezterm.action_callback(function(_, inner_pane, id, _)
            if id then
              wezterm.run_child_process {
                wezterm_binary(),
                'cli', 'split-pane',
                '--move-pane-id', id,
                '--pane-id', tostring(inner_pane:pane_id()),
              }
            end
          end),
        },
        pane
      )
    end),
  },
  {
    key = 'h',
    mods = 'LEADER',
    action = wezterm.action_callback(function(window, pane)
      local mux_window = window:mux_window()
      if not mux_window then
        return
      end

      local choices = build_merge_tab_choices(mux_window)
      if #choices == 0 then
        window:toast_notification('merge tab', 'No other tabs to merge', nil, 3000)
        return
      end

      window:perform_action(
        act.InputSelector {
          title = "Select Tab to Convert into a Split Pane",
          fuzzy = true,
          choices = choices,
          action = wezterm.action_callback(function(_, inner_pane, id, _)
            if id then
              wezterm.run_child_process {
                wezterm_binary(),
                'cli', 'split-pane', '--horizontal',
                '--move-pane-id', id,
                '--pane-id', tostring(inner_pane:pane_id()),
              }
            end
          end),
        },
        pane
      )
    end),
  },
  
  -- Close the current pane
  {
    key = 'x',
    mods = 'LEADER',
    action = wezterm.action.CloseCurrentPane { confirm = true },
  },
  -- Change the title of the current tab
  {
    key = 'e',
    mods = 'LEADER',
    action = act.PromptInputLine {
      description = 'Enter new name for tab',
      action = wezterm.action_callback(
        function(window, _, line)
          if line then
            window:active_tab():set_title(line)
          end
        end
      ),
    },
  },
}

-- Return the configuration to wezterm
return config
