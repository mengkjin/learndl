// 获取画布和上下文
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

// 定义网格大小和蛇的初始位置
const gridSize = 20;
const tileCount = canvas.width / gridSize;
let snake = [
    {x: 10, y: 10},
];
let food = {
    x: Math.floor(Math.random() * tileCount),
    y: Math.floor(Math.random() * tileCount)
};

// 初始速度
let xVelocity = 0;
let yVelocity = 0;

// 游戏状态变量
let score = 0;
let highScore = localStorage.getItem('snakeHighScore') || 0;
let gameSpeed = 100;
let isGamePaused = false;
let gameInterval;
let isGameRunning = false;

// 获取DOM元素
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const difficultySelect = document.getElementById('difficulty');
const scoreText = document.getElementById('scoreText');
const highScoreText = document.getElementById('highScoreText');

// 更新界面分数显示
function updateScores() {
    scoreText.textContent = score;
    highScoreText.textContent = highScore;
}

// 设置游戏速度
function setGameSpeed() {
    switch(difficultySelect.value) {
        case 'easy':
            gameSpeed = 120;
            break;
        case 'medium':
            gameSpeed = 100;
            break;
        case 'hard':
            gameSpeed = 70;
            break;
    }
}

// 重置游戏
function resetGame() {
    snake = [{x: 10, y: 10}];
    xVelocity = 0;
    yVelocity = 0;
    score = 0;
    updateScores();
    generateFood();
}

// 生成食物（确保不会出现在蛇身上）
function generateFood() {
    let newFood;
    do {
        newFood = {
            x: Math.floor(Math.random() * tileCount),
            y: Math.floor(Math.random() * tileCount)
        };
    } while (snake.some(segment => segment.x === newFood.x && segment.y === newFood.y));
    food = newFood;
}

// 修改游戏主循环
function gameLoop() {
    if (!isGamePaused && isGameRunning) {
        moveSnake();
        if (checkGameOver()) {
            handleGameOver();
            return;
        }
        drawGame();
    }
    setTimeout(gameLoop, gameSpeed);
}

// 处理游戏结束
function handleGameOver() {
    if (score > highScore) {
        highScore = score;
        localStorage.setItem('snakeHighScore', highScore);
    }
    updateScores();
    alert(`游戏结束！\n你的得分：${score}\n最高分：${highScore}`);
    isGameRunning = false;
    startBtn.textContent = '重新开始';
}

// 修改移动蛇的函数
function moveSnake() {
    const head = {x: snake[0].x + xVelocity, y: snake[0].y + yVelocity};
    snake.unshift(head);
    
    if (head.x === food.x && head.y === food.y) {
        score += 10;
        updateScores();
        generateFood();
    } else {
        snake.pop();
    }
}

// 修改绘制函数，添加网格背景
function drawGame() {
    // 清空画布
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 绘制网格
    ctx.strokeStyle = '#ddd';
    for (let i = 0; i < tileCount; i++) {
        ctx.beginPath();
        ctx.moveTo(i * gridSize, 0);
        ctx.lineTo(i * gridSize, canvas.height);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, i * gridSize);
        ctx.lineTo(canvas.width, i * gridSize);
        ctx.stroke();
    }
    
    // 绘制蛇
    snake.forEach((segment, index) => {
        ctx.fillStyle = index === 0 ? '#2ecc71' : '#27ae60';
        ctx.fillRect(segment.x * gridSize, segment.y * gridSize, gridSize - 2, gridSize - 2);
    });
    
    // 绘制食物
    ctx.fillStyle = '#e74c3c';
    ctx.beginPath();
    ctx.arc(
        food.x * gridSize + gridSize/2,
        food.y * gridSize + gridSize/2,
        gridSize/2 - 2,
        0,
        Math.PI * 2
    );
    ctx.fill();
}

// 检查游戏是否结束
function checkGameOver() {
    const head = snake[0];
    // 检查是否撞墙
    if (head.x < 0 ||
        head.x >= tileCount ||
        head.y < 0 ||
        head.y >= tileCount) {
        return true;
    }
    
    // 检查是否撞到自己
    for (let i = 1; i < snake.length; i++) {
        if (head.x === snake[i].x && head.y === snake[i].y) {
            return true;
        }
    }
    
    return false;
}

// 添加键盘控制（这部分也似乎缺失了）
document.addEventListener('keydown', function(event) {
    switch(event.key) {
        case 'ArrowUp':
            if (yVelocity !== 1 && isGameRunning && !isGamePaused) {
                xVelocity = 0;
                yVelocity = -1;
            }
            break;
        case 'ArrowDown':
            if (yVelocity !== -1 && isGameRunning && !isGamePaused) {
                xVelocity = 0;
                yVelocity = 1;
            }
            break;
        case 'ArrowLeft':
            if (xVelocity !== 1 && isGameRunning && !isGamePaused) {
                xVelocity = -1;
                yVelocity = 0;
            }
            break;
        case 'ArrowRight':
            if (xVelocity !== -1 && isGameRunning && !isGamePaused) {
                xVelocity = 1;
                yVelocity = 0;
            }
            break;
    }
});

// 事件监听器
startBtn.addEventListener('click', () => {
    if (!isGameRunning) {
        resetGame();
        isGameRunning = true;
        isGamePaused = false;
        startBtn.textContent = '重新开始';
        pauseBtn.textContent = '暂停';
    } else {
        if (confirm('确定要重新开始吗？')) {
            resetGame();
        }
    }
});

pauseBtn.addEventListener('click', () => {
    if (isGameRunning) {
        isGamePaused = !isGamePaused;
        pauseBtn.textContent = isGamePaused ? '继续' : '暂停';
    }
});

difficultySelect.addEventListener('change', setGameSpeed);

// 初始化显示
updateScores();
setGameSpeed();
gameLoop();
