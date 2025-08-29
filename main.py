import os
import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from telegram.constants import ParseMode
import sqlite3
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class DostyqTVBot:
    def __init__(self):
        # Конфигурация из .env файла
        self.BOT_TOKEN = os.getenv('BOT_TOKEN')
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        self.ADMIN_IDS = [int(x) for x in os.getenv('ADMIN_IDS', '').split(',') if x]

        # База знаний DostyqTV
        self.knowledge_base = {
            'программы': {
                'новости': 'Новости запустятся в ближайшее время. Следите за обновлениями!',
                'сериалы': 'Популярные сериалы: "Троцкий", "Детективный синдром", "Актер"',
                'детские': 'Детские программы запустятся в ближайшее время. Следите за обновлениями!',
                'спорт': 'Спортивные программы запустятся в ближайшее время. Следите за обновлениями!'
            },
            'технические': {
                'качество': 'Вещание в HD качестве 1080p',
                'каналы': 'DostyqTV доступен на позиции 44 в кабельных сетях',
                'интернет': 'Онлайн трансляция доступна на сайте dostyq.tv',
                'приложение': 'Мобильное приложение DostyqTV в App Store и Google Play'
            },
            'контакты': {
                'телефон': '+7 777 013 3812',
                'email': 'support@dostyq.tv',
                'адрес': 'https://dostyq.tv',
                'сайт': 'https://dostyq.tv'
            }
        }

        # FAQ база
        self.faq = {
            'Как настроить канал?': 'Для настройки канала обратитесь к оператору кабельного ТВ или найдите канал DostyqTV на позиции 44',
            'Проблемы с качеством': 'Проверьте силу сигнала, перезагрузите приставку. Если проблема остается - звоните +7 771 300 05 02, +7 702 300 05 01',
            'Расписание программ': 'Полное расписание доступно на сайте dostyq.tv в разделе "Каталог"',
            'Реклама на канале': 'По вопросам размещения рекламы: reklama@dostyq.tv или +7 (727) 24-24-25'
        }

    async def init_db(self):
        """Инициализация базы данных"""
        conn = sqlite3.connect('dostyqtv_bot.db')
        cursor = conn.cursor()

        # Таблица пользователей
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Таблица обращений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT,
                status TEXT DEFAULT 'open',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                category TEXT
            )
        ''')

        # Таблица статистики
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stats (
                date DATE PRIMARY KEY,
                users_count INTEGER DEFAULT 0,
                messages_count INTEGER DEFAULT 0,
                tickets_count INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    async def get_ai_response(self, user_message: str, user_context: Dict = None) -> str:
        """Получение ответа от AI API Gemini"""
        if not self.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY не найден. Используется резервный ответ.")
            return await self.get_fallback_response(user_message)

        try:
            system_prompt = f"""
            Ты - помощник службы поддержки телеканала DostyqTV. Отвечай на казахском или русском языке в зависимости от языка вопроса.

            База знаний:
            {json.dumps(self.knowledge_base, ensure_ascii=False, indent=2)}

            FAQ:
            {json.dumps(self.faq, ensure_ascii=False, indent=2)}

            Правила:
            1. Всегда будь вежливым и профессиональным.
            2. Если не знаешь точного ответа, предложи обратиться в службу поддержки, указав контакты.
            3. Используй эмодзи для улучшения восприятия.
            4. Ответы должны быть краткими, но информативными.
            5. При технических проблемах предлагай пошаговые решения.
            """

            return await self._call_gemini(user_message, system_prompt)

        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            return await self.get_fallback_response(user_message)

    async def _call_gemini(self, user_message: str, system_prompt: str) -> str:
        """Google Gemini API вызов"""
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.GEMINI_API_KEY}'

        headers = {'Content-Type': 'application/json'}

        data = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt}\n\nПользователь: {user_message}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 512
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    # Проверка на наличие контента в ответе
                    if 'candidates' in result and result['candidates']:
                        return result['candidates'][0]['content']['parts'][0]['text']
                    else:
                        logger.error(f"Gemini API response format error: {result}")
                        return await self.get_fallback_response(user_message)
                else:
                    error_text = await response.text()
                    logger.error(f"Gemini API error: {response.status}, {error_text}")
                    return await self.get_fallback_response(user_message)

    async def get_fallback_response(self, user_message: str) -> str:
        """Резервные ответы на основе ключевых слов"""
        message_lower = user_message.lower()

        # Поиск по ключевым словам
        if any(word in message_lower for word in ['программа', 'расписание', 'передач']):
            return "📺 Полное расписание программ доступно на сайте dostyq.tv в разделе 'Каталог'"

        elif any(word in message_lower for word in ['качество', 'плохо', 'тормозит', 'зависает']):
            return "🔧 При проблемах с качеством, попробуйте перезагрузить приставку. Если не помогло, свяжитесь с поддержкой: +7 771 300 05 02, +7 702 300 05 01"

        elif any(word in message_lower for word in ['настроить', 'канал', 'найти']):
            return "⚙️ Настройка канала DostyqTV:\n\n📍 Найдите нас на позиции 44 в кабельных сетях. Если не получается, обратитесь к вашему оператору кабельного ТВ."

        elif any(word in message_lower for word in ['контакт', 'телефон', 'связаться']):
            return "📞 Контакты DostyqTV:\n\n☎️ Поддержка: +7 777 013 3812\n📧 Email: support@dostyq.tv\n🌐 Сайт: https://dostyq.tv"

        elif any(word in message_lower for word in ['реклама', 'размещение']):
            return "📢 По вопросам размещения рекламы:\n\n📧 reklama@dostyq.tv\n☎️ +7 (727) 24-24-25"

        else:
            return "👋 Здравствуйте! Я помогу вам с вопросами по телеканалу DostyqTV.\n\n🔍 Используйте /help для просмотра доступных команд или просто опишите вашу проблему."

    async def save_user(self, user):
        """Сохранение информации о пользователе"""
        conn = sqlite3.connect('dostyqtv_bot.db')
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO users (id, username, first_name, last_name, last_activity)
            VALUES (?, ?, ?, ?, ?)
        ''', (user.id, user.username, user.first_name, user.last_name, datetime.now()))

        conn.commit()
        conn.close()

    async def create_ticket(self, user_id: int, message: str, category: str = 'general'):
        """Создание тикета"""
        conn = sqlite3.connect('dostyqtv_bot.db')
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO tickets (user_id, message, category)
            VALUES (?, ?, ?)
        ''', (user_id, message, category))

        ticket_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return ticket_id

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user
        await self.save_user(user)

        welcome_message = f"""
👋 Добро пожаловать в службу поддержки DostyqTV!

Привет, {user.first_name}! Я ваш виртуальный помощник. Помогу решить любые вопросы связанные с просмотром DostyqTV.

🔥 Что я умею:
• Отвечать на вопросы о программах передач
• Помогать с техническими проблемами
• Предоставлять контактную информацию
• Создавать обращения в службу поддержки

💡 Просто напишите ваш вопрос или используйте команды ниже:
        """

        keyboard = [
            [InlineKeyboardButton("📺 Программа передач", callback_data='schedule')],
            [InlineKeyboardButton("🔧 Техническая поддержка", callback_data='tech_support')],
            [InlineKeyboardButton("📞 Контакты", callback_data='contacts')],
            [InlineKeyboardButton("📋 FAQ", callback_data='faq')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(welcome_message, reply_markup=reply_markup)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
🤖 Команды бота DostyqTV:

/start - Начать работу с ботом
/help - Показать это сообщение
/schedule - Программа передач
/contact - Контактная информация
/faq - Часто задаваемые вопросы
/ticket - Создать обращение в поддержку
/status - Проверить статус обращения

💬 Вы также можете просто написать ваш вопрос, и я постараюсь помочь!
        """

        await update.message.reply_text(help_text)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик inline кнопок"""
        query = update.callback_query
        await query.answer()

        data = query.data

        if data == 'schedule':
            schedule_text = self.faq['Расписание программ']
            await query.edit_message_text(f"📅 Расписание программ:\n\n{schedule_text}")

        elif data == 'tech_support':
            tech_text = self.faq['Проблемы с качеством']
            await query.edit_message_text(f"🔧 Техническая поддержка:\n\n{tech_text}")

        elif data == 'contacts':
            contacts_text = f"📞 Контакты:\n\nТелефон: {self.knowledge_base['контакты']['телефон']}\nEmail: {self.knowledge_base['контакты']['email']}\nСайт: {self.knowledge_base['контакты']['сайт']}"
            await query.edit_message_text(contacts_text)

        elif data == 'faq':
            faq_items = [f"❓ {q}\n✅ {a}" for q, a in self.faq.items()]
            faq_text = "\n\n".join(faq_items)
            await query.edit_message_text(f"📋 Часто задаваемые вопросы:\n\n{faq_text}")

        elif data == 'create_ticket':
            await self.ticket_command(update, context)


    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        user = update.effective_user
        await self.save_user(user)

        user_message = update.message.text

        # Показываем индикатор печатания
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

        # Получаем ответ от AI или fallback
        response = await self.get_ai_response(user_message)

        # Добавляем кнопки быстрых действий
        keyboard = [
            [InlineKeyboardButton("📞 Связаться с поддержкой", callback_data='contacts')],
            [InlineKeyboardButton("📋 Создать обращение", callback_data='create_ticket')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(response, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)

    async def ticket_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда создания тикета"""
        # Эта функция может быть расширена для сбора деталей тикета
        user_id = update.effective_user.id
        # Для простоты создаем тикет с последним сообщением пользователя
        # В реальном приложении логика будет сложнее
        message_text = "Пользователь запросил создание тикета."

        ticket_id = await self.create_ticket(user_id, message_text)

        response_text = (
            f"📋 Ваше обращение #{ticket_id} создано!\n\n"
            "Наши специалисты скоро свяжутся с вами. "
            "Пожалуйста, опишите вашу проблему подробнее в следующем сообщении."
        )

        # Если команда вызвана из callback кнопки
        if update.callback_query:
            await update.callback_query.edit_message_text(response_text)
        else: # Если команда вызвана через /ticket
            await update.message.reply_text(response_text)


    async def get_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика для администраторов"""
        if update.effective_user.id not in self.ADMIN_IDS:
            await update.message.reply_text("❌ У вас нет прав для просмотра статистики")
            return

        conn = sqlite3.connect('dostyqtv_bot.db')
        cursor = conn.cursor()

        # Общая статистика
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM tickets WHERE status = "open"')
        open_tickets = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM tickets WHERE date(created_at) = date("now")')
        today_tickets = cursor.fetchone()[0]

        stats_text = f"""
📊 Статистика DostyqTV Bot:

👥 Пользователи: {total_users}
🎫 Открытые обращения: {open_tickets}
📋 Обращения за сегодня: {today_tickets}

📅 Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}
        """

        conn.close()
        await update.message.reply_text(stats_text)

    def setup_handlers(self, application):
        """Настройка обработчиков"""
        # Команды
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("ticket", self.ticket_command))
        application.add_handler(CommandHandler("stats", self.get_stats))

        # Callback кнопки
        application.add_handler(CallbackQueryHandler(self.handle_callback))

        # Текстовые сообщения
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def set_bot_commands(self, application):
        """Установка команд бота"""
        commands = [
            BotCommand("start", "🚀 Начать работу с ботом"),
            BotCommand("help", "❓ Помощь и список команд"),
            BotCommand("schedule", "📺 Программа передач"),
            BotCommand("contact", "📞 Контактная информация"),
            BotCommand("faq", "❓ Часто задаваемые вопросы"),
            BotCommand("ticket", "📋 Создать обращение")
        ]

        await application.bot.set_my_commands(commands)
    
    # ИЗМЕНЕНИЕ 1: Создаем новую асинхронную функцию для задач, которые нужно выполнить ДО запуска бота
    async def post_init(self, application: Application):
        """Задачи, выполняемые после инициализации приложения, но до запуска опроса."""
        await self.init_db()
        await self.set_bot_commands(application)

    def run(self):
        """Запуск бота"""
        if not self.BOT_TOKEN:
            logger.error("BOT_TOKEN не установлен! Проверьте ваш .env файл.")
            return
        if not self.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY не установлен! Проверьте ваш .env файл.")
            return

        # ИЗМЕНЕНИЕ 2: Используем специальный параметр `post_init`, чтобы безопасно выполнить наш асинхронный код
        application = (
            Application.builder()
            .token(self.BOT_TOKEN)
            .post_init(self.post_init)
            .build()
        )

        # Настройка обработчиков
        self.setup_handlers(application)
        
        logger.info("DostyqTV Support Bot запущен!")
        
        # ИЗМЕНЕНИЕ 3: `run_polling` теперь вызывается напрямую. Он сам управляет циклом событий.
        application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    bot = DostyqTVBot()
    # ИЗМЕНЕНИЕ 4: Убираем `asyncio.run()`. Просто вызываем синхронный метод `run`.
    bot.run()