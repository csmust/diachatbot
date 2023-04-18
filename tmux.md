# 创建新会话

```tmux new -s session_name```



# 创建会话中的新窗口
``` 第一步：按 Ctrl+B 组合键，然后松开。```

``` 第二步：再单独按一下 c 键。```

# 切换窗口
```很简单，假如我们要切换到 1：bash 这个窗口，步骤如下：```

```第一步：按 Ctrl-B 组合键，然后松开。```

``` 第二步：按数字 1 键。```

# 离开电脑

第一步：输入组合键 Ctrl+B，然后松开。

第二步：输入字母 d。

# 重新链接电脑
```tmux ls```

```tmux a -t session_name```

杀死某个会话：
tmux kill-session -t {session-name} 杀死某个 session。
